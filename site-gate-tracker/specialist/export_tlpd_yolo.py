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
        with urllib.request.urlopen(req,timeout=30) as r:
            meta=json.load(r)
        urls=[x["url"] for x in meta.get("parquet_files",[]) if x.get("split")=="train" and x.get("url")]
        if not urls:
            raise RuntimeError("Hugging Face datasets-server returned no TLPD train parquet files") from first
        print(json.dumps({"tlpd_parquet_files":len(urls)},indent=2),flush=True)
        return load_dataset("parquet",data_files={"train":urls},split="train")

def decode_points(v):
    if isinstance(v,str):
        try:v=json.loads(v)
        except:return []
    if isinstance(v,list) and v and isinstance(v[0],list):return v
    return []

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--seed",type=int,default=260918);args=ap.parse_args()
    ds=load_tlpd()
    order=list(range(len(ds)));random.Random(args.seed).shuffle(order)
    n=len(order);cuts=(int(n*.80),int(n*.90));manifest=[]
    for rank,idx in enumerate(order):
        row=ds[idx];im=row.get("image")
        if not isinstance(im,Image.Image):
            try:im=Image.fromarray(im)
            except:continue
        im=im.convert("RGB");W,H=im.size
        pts=decode_points(row.get("points"))
        if not pts:continue
        xs=[float(p[0]) for p in pts];ys=[float(p[1]) for p in pts]
        x1,x2=max(0,min(xs)),min(W,max(xs));y1,y2=max(0,min(ys)),min(H,max(ys))
        bw=x2-x1;bh=y2-y1
        if bw<3 or bh<3:continue
        split="train" if rank<cuts[0] else ("val" if rank<cuts[1] else "test")
        od=args.out/split;(od/"images").mkdir(parents=True,exist_ok=True);(od/"labels").mkdir(parents=True,exist_ok=True)
        fn=f"tlpd_{rank:05d}.jpg";im.save(od/"images"/fn,quality=92)
        lab=f"0 {(x1+x2)/(2*W):.6f} {(y1+y2)/(2*H):.6f} {bw/W:.6f} {bh/H:.6f}\n"
        (od/"labels"/(Path(fn).stem+".txt")).write_text(lab)
        manifest.append({"file":f"{split}/images/{fn}","source":REPO,"license":"MIT"})
    (args.out/"data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest))
    print(json.dumps({"samples":len(manifest),"source":REPO,"license":"MIT"},indent=2))
    if len(manifest)<2500:raise SystemExit(f"TLPD export unexpectedly small: {len(manifest)}")
if __name__=="__main__":main()
