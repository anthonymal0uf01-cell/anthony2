#!/usr/bin/env python3
"""
Build a large, licence-auditable plate-localization pretraining corpus.

Sources:
  - CCPD mirror on Hugging Face (large-scale localization geometry)
  - TLPD (MIT, polygon annotations)
  - Indian Vehicle License Plate Localization (CC BY 4.0, YOLO)

This is localization-only. Plate text is not used.

Important:
  Dataset-card licensing is recorded verbatim. Upstream rights must still be
  verified before commercial redistribution/promotion. The builder therefore
  writes provenance for every sample and never merges provenance away.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, random, shutil, tarfile, zipfile
from pathlib import Path
from typing import Any

from PIL import Image

def sha1(s:str)->str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def split_for(key:str)->str:
    v=int(sha1(key)[:8],16)%100
    return "train" if v<82 else ("val" if v<92 else "test")

def ensure_dirs(root:Path):
    for s in ("train","val","test"):
        (root/s/"images").mkdir(parents=True,exist_ok=True)
        (root/s/"labels").mkdir(parents=True,exist_ok=True)

def pil_from(v:Any)->Image.Image|None:
    if isinstance(v,Image.Image): return v.convert("RGB")
    if isinstance(v,(bytes,bytearray)):
        try:return Image.open(io.BytesIO(v)).convert("RGB")
        except Exception:return None
    if isinstance(v,dict):
        b=v.get("bytes")
        p=v.get("path")
        if b:
            try:return Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:pass
        if p and Path(p).exists():
            try:return Image.open(p).convert("RGB")
            except Exception:pass
    return None

def find_image(sample:dict)->Image.Image|None:
    for k in ("image","jpg","jpeg","png","webp"):
        if k in sample:
            im=pil_from(sample[k])
            if im:return im
    for v in sample.values():
        im=pil_from(v)
        if im:return im
    return None

def parse_ccpd_bbox(key:str):
    name=Path(key).name
    parts=name.split("-")
    if len(parts)<4:return None
    try:
        a,b=parts[2].split("_")[:2]
        x1,y1=map(int,a.split("&"))
        x2,y2=map(int,b.split("&"))
        x1,x2=sorted((x1,x2));y1,y2=sorted((y1,y2))
        if x2<=x1 or y2<=y1:return None
        return [x1,y1,x2,y2]
    except Exception:return None

def yolo_line(box,W,H):
    x1,y1,x2,y2=box
    x1=max(0,min(W-1,float(x1)));x2=max(0,min(W,float(x2)))
    y1=max(0,min(H-1,float(y1)));y2=max(0,min(H,float(y2)))
    if x2<=x1 or y2<=y1:return None
    xc=(x1+x2)/2/W;yc=(y1+y2)/2/H
    bw=(x2-x1)/W;bh=(y2-y1)/H
    return f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"

class Writer:
    def __init__(self,root:Path):
        self.root=root;ensure_dirs(root)
        self.seen=set();self.manifest=[];self.counts={}
    def write(self,source,key,im,boxes,license_name,policy,extra=None):
        buf=io.BytesIO();im.save(buf,format="JPEG",quality=92,subsampling=0)
        raw=buf.getvalue();h=hashlib.sha256(raw).hexdigest()
        if h in self.seen:return False
        W,H=im.size
        labs=[yolo_line(b,W,H) for b in boxes]
        labs=[x for x in labs if x]
        if not labs:return False
        split=split_for(source+":"+key)
        stem=f"{source}_{sha1(key)[:18]}"
        ip=self.root/split/"images"/(stem+".jpg")
        lp=self.root/split/"labels"/(stem+".txt")
        ip.write_bytes(raw);lp.write_text("\n".join(labs)+"\n",encoding="utf-8")
        self.seen.add(h)
        self.counts[source]=self.counts.get(source,0)+1
        self.manifest.append({
          "file":f"{split}/images/{ip.name}","source":source,"source_key":key,
          "license":license_name,"production_policy":policy,"boxes":boxes,
          "image_size":[W,H],"sha256":h,**(extra or {})
        })
        return True
    def finish(self):
        (self.root/"manifest.jsonl").write_text(
            "\n".join(json.dumps(x,ensure_ascii=False) for x in self.manifest),
            encoding="utf-8")
        (self.root/"data.yaml").write_text(
            f"path: {self.root.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n",
            encoding="utf-8")
        split_counts={}
        for s in ("train","val","test"):
            split_counts[s]=len(list((self.root/s/"images").glob("*.jpg")))
        summary={"total":len(self.manifest),"by_source":self.counts,"by_split":split_counts}
        (self.root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(json.dumps(summary))

def add_ccpd(w:Writer,limit:int,seed:int):
    if limit<=0:return
    from datasets import load_dataset
    ds=load_dataset("JorgeLlorente/CCPD-Dataset",split="train",streaming=True)
    try:ds=ds.shuffle(seed=seed,buffer_size=min(5000,max(1000,limit//2)))
    except Exception:pass
    n=0
    for sample in ds:
        if n>=limit:break
        key=str(sample.get("__key__") or sample.get("key") or sample.get("__url__") or n)
        box=parse_ccpd_bbox(key)
        im=find_image(sample)
        if not box or im is None:continue
        if w.write(
          "ccpd_hf",key,im,[box],"apache-2.0 (HF dataset card)",
          "pretraining_review_upstream_rights_before_commercial_promotion",
          {"dataset":"JorgeLlorente/CCPD-Dataset",
           "note":"HF mirror attributes original CCPD creators; upstream rights should be independently verified."}
        ): n+=1

def snapshot(repo_id:str):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo_id,repo_type="dataset"))

def add_tlpd(w:Writer,limit:int):
    if limit<=0:return
    root=snapshot("evan6007/TLPD")
    labels=root/"labels";images=root/"images"
    n=0
    for jp in sorted(labels.glob("*.json")):
        if n>=limit:break
        try:j=json.loads(jp.read_text(encoding="utf-8"))
        except Exception:continue
        boxes=[]
        for sh in j.get("shapes",[]):
            pts=sh.get("points") or []
            if len(pts)<2:continue
            xs=[float(p[0]) for p in pts];ys=[float(p[1]) for p in pts]
            boxes.append([min(xs),min(ys),max(xs),max(ys)])
        if not boxes:continue
        ip=None
        for ext in (".jpg",".jpeg",".png",".webp"):
            p=images/(jp.stem+ext)
            if p.exists():ip=p;break
        if ip is None:continue
        try:im=Image.open(ip).convert("RGB")
        except Exception:continue
        if w.write(
          "tlpd",jp.stem,im,boxes,"MIT","commercial_ok_with_attribution",
          {"dataset":"evan6007/TLPD","annotation":"LabelMe polygon→bbox"}
        ):n+=1

def extract_archives(root:Path):
    out=root/"_extracted";out.mkdir(exist_ok=True)
    for p in list(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(p) as z:z.extractall(out/p.stem)
        except Exception:pass
    for pattern in ("*.tar.gz","*.tgz","*.tar"):
        for p in list(root.rglob(pattern)):
            try:
                with tarfile.open(p) as t:t.extractall(out/p.name.replace(".tar.gz","").replace(".tgz","").replace(".tar",""))
            except Exception:pass
    return root

def add_indian(w:Writer,limit:int):
    if limit<=0:return
    root=extract_archives(snapshot("thundarstrom/indian-license-plate-detection"))
    n=0
    for lp in root.rglob("*.txt"):
        if n>=limit:break
        if "labels" not in lp.parts:continue
        txt=lp.read_text(encoding="utf-8",errors="ignore").strip().splitlines()
        rows=[]
        for line in txt:
            p=line.split()
            if len(p)<5:continue
            try:
                _,xc,yc,bw,bh=map(float,p[:5])
            except Exception:continue
            rows.append((xc,yc,bw,bh))
        if not rows:continue
        # Match the common YOLO images/labels directory relation.
        s=str(lp)
        candidates=[]
        if os.sep+"labels"+os.sep in s:
            base=s.replace(os.sep+"labels"+os.sep,os.sep+"images"+os.sep)
            for ext in (".jpg",".jpeg",".png",".webp"):
                candidates.append(Path(base).with_suffix(ext))
        candidates += [lp.with_suffix(x) for x in (".jpg",".jpeg",".png",".webp")]
        ip=next((p for p in candidates if p.exists()),None)
        if ip is None:continue
        try:im=Image.open(ip).convert("RGB")
        except Exception:continue
        W,H=im.size;boxes=[]
        for xc,yc,bw,bh in rows:
            boxes.append([(xc-bw/2)*W,(yc-bh/2)*H,(xc+bw/2)*W,(yc+bh/2)*H])
        key=str(lp.relative_to(root))
        if w.write(
          "indian_hf",key,im,boxes,"CC BY 4.0","commercial_ok_with_attribution",
          {"dataset":"thundarstrom/indian-license-plate-detection","annotation":"YOLO bbox"}
        ):n+=1

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--ccpd-limit",type=int,default=12000)
    ap.add_argument("--tlpd-limit",type=int,default=3032)
    ap.add_argument("--indian-limit",type=int,default=3742)
    ap.add_argument("--seed",type=int,default=260922)
    a=ap.parse_args();random.seed(a.seed)
    w=Writer(a.out)
    add_ccpd(w,a.ccpd_limit,a.seed)
    add_tlpd(w,a.tlpd_limit)
    add_indian(w,a.indian_limit)
    w.finish()

if __name__=="__main__":main()
