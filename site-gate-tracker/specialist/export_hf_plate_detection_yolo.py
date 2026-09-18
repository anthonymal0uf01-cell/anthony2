#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from PIL import Image

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=1200)
    args=ap.parse_args()
    from datasets import load_dataset
    ds=load_dataset("justjuu/license-plate-detection")
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    split_map={"train":"train","validation":"val","test":"test"}
    used=0
    for src_split,dst_split in split_map.items():
        if src_split not in ds: continue
        imgs=args.out/dst_split/"images"; labs=args.out/dst_split/"labels"
        imgs.mkdir(parents=True,exist_ok=True);labs.mkdir(parents=True,exist_ok=True)
        rows=ds[src_split]
        cap=args.limit if src_split=="train" else min(250,len(rows))
        for i,row in enumerate(rows.select(range(min(cap,len(rows))))):
            im=row["image"]
            if not isinstance(im,Image.Image): im=Image.fromarray(im)
            im=im.convert("RGB")
            w,h=im.size
            boxes=(row.get("objects") or {}).get("bbox") or []
            y=[]
            for b in boxes:
                x0,y0,bw,bh=map(float,b)
                if bw<3 or bh<3: continue
                xc=(x0+bw/2)/w;yc=(y0+bh/2)/h
                y.append(f"0 {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}")
            if not y: continue
            fn=f"hf_{src_split}_{i:06d}.jpg"
            im.save(imgs/fn,quality=92)
            (labs/(Path(fn).stem+".txt")).write_text("\n".join(y)+"\n")
            manifest.append({"split":dst_split,"file":fn,"source":"justjuu/license-plate-detection","license":"CC BY 4.0"})
            used+=1
    (args.out/"data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n",
        encoding="utf-8")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest),encoding="utf-8")
    print(json.dumps({"samples":used,"source":"justjuu/license-plate-detection","license":"CC BY 4.0"},indent=2))
if __name__=="__main__":main()
