#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random,urllib.parse,urllib.request
from pathlib import Path
from PIL import Image

REPO="evan6007/TLPD"

def load_tlpd():
    from datasets import load_dataset
    try:
        return load_dataset(REPO,split="train")
    except Exception as first:
        print("DIRECT_TLPD_LOAD_FAILED",repr(first),flush=True)
        api="https://datasets-server.huggingface.co/parquet?dataset="+urllib.parse.quote(REPO,safe="")
        req=urllib.request.Request(api,headers={"User-Agent":"SiteGatePhase2/2026"})
        with urllib.request.urlopen(req,timeout=30) as r: meta=json.load(r)
        urls=[x["url"] for x in meta.get("parquet_files",[]) if x.get("split")=="train" and x.get("url")]
        if not urls: raise RuntimeError("No TLPD parquet files from datasets-server") from first
        print(json.dumps({"tlpd_parquet_files":len(urls)},indent=2),flush=True)
        return load_dataset("parquet",data_files={"train":urls},split="train")

def _numeric_pair(x):
    return isinstance(x,(list,tuple)) and len(x)>=2 and all(isinstance(v,(int,float)) for v in x[:2])

def find_points(node):
    # LabelMe points may survive auto-conversion as nested lists, dicts or JSON strings.
    if isinstance(node,str):
        s=node.strip()
        if s[:1] in "[{":
            try:return find_points(json.loads(s))
            except: return []
        return []
    if isinstance(node,dict):
        for key in ("points","polygon","vertices","segmentation"):
            if key in node:
                p=find_points(node[key])
                if p:return p
        for v in node.values():
            p=find_points(v)
            if p:return p
        return []
    if isinstance(node,(list,tuple)):
        if len(node)>=3 and all(_numeric_pair(x) for x in node):
            return [[float(x[0]),float(x[1])] for x in node]
        for x in node:
            p=find_points(x)
            if p:return p
    return []

def to_image(v):
    if isinstance(v,Image.Image): return v.convert("RGB")
    if isinstance(v,dict):
        if v.get("bytes"):
            import io
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        if v.get("path"):
            return Image.open(v["path"]).convert("RGB")
    try:return Image.fromarray(v).convert("RGB")
    except:return None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--seed",type=int,default=260918);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    ds=load_tlpd()
    if len(ds):
        first=ds[0]
        print("TLPD_COLUMNS",list(first.keys()),flush=True)
        print("TLPD_TYPES",{k:type(v).__name__ for k,v in first.items()},flush=True)
    order=list(range(len(ds)));random.Random(args.seed).shuffle(order)
    n=len(order);cuts=(int(n*.80),int(n*.90));manifest=[];miss=0
    for rank,idx in enumerate(order):
        row=ds[idx];im=to_image(row.get("image"))
        if im is None: miss+=1;continue
        W,H=im.size
        pts=find_points(row)
        if not pts:
            miss+=1;continue
        xs=[float(p[0]) for p in pts];ys=[float(p[1]) for p in pts]
        x1,x2=max(0,min(xs)),min(W,max(xs));y1,y2=max(0,min(ys)),min(H,max(ys))
        bw=x2-x1;bh=y2-y1
        if bw<3 or bh<3:miss+=1;continue
        split="train" if rank<cuts[0] else ("val" if rank<cuts[1] else "test")
        od=args.out/split;(od/"images").mkdir(parents=True,exist_ok=True);(od/"labels").mkdir(parents=True,exist_ok=True)
        fn=f"tlpd_{rank:05d}.jpg";im.save(od/"images"/fn,quality=92)
        (od/"labels"/(Path(fn).stem+".txt")).write_text(
            f"0 {(x1+x2)/(2*W):.6f} {(y1+y2)/(2*H):.6f} {bw/W:.6f} {bh/H:.6f}\n")
        manifest.append({"file":f"{split}/images/{fn}","source":REPO,"license":"MIT"})
    (args.out/"data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest))
    print(json.dumps({"samples":len(manifest),"missed":miss,"source":REPO,"license":"MIT"},indent=2))
    if len(manifest)<2400:raise SystemExit(f"TLPD export unexpectedly small: {len(manifest)} / {len(ds)}")
if __name__=="__main__":main()
