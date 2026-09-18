#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image

SPLITS=("train","valid","val","test")

def ahash(path:Path,size=16):
    with Image.open(path) as im:
        im=im.convert("L").resize((size,size),Image.Resampling.BILINEAR)
        a=np.asarray(im,dtype=np.float32)
    return "".join("1" if x>a.mean() else "0" for x in a.ravel())

def hamming(a,b): return sum(x!=y for x,y in zip(a,b))

def main():
    ap=argparse.ArgumentParser(description="Merge/dedupe Australian YOLO plate datasets into one 1-class corpus.")
    ap.add_argument("sources",nargs="+",type=Path)
    ap.add_argument("--out",type=Path,default=Path("data/au_plate_merged"))
    ap.add_argument("--phash-distance",type=int,default=3)
    args=ap.parse_args()
    for s in ("train","val","test"):
        (args.out/s/"images").mkdir(parents=True,exist_ok=True)
        (args.out/s/"labels").mkdir(parents=True,exist_ok=True)
    seen=[];manifest=[];n=0;dupes=0
    for source in args.sources:
        for split in SPLITS:
            imgdir=source/split/"images"
            labdir=source/split/"labels"
            if not imgdir.exists(): continue
            outsplit="val" if split in {"valid","val"} else split
            for img in sorted(imgdir.iterdir()):
                if img.suffix.lower() not in {".jpg",".jpeg",".png",".webp"}: continue
                h=ahash(img)
                if any(hamming(h,x)<=args.phash_distance for x in seen):
                    dupes+=1;continue
                seen.append(h);label=labdir/(img.stem+".txt")
                if not label.exists(): continue
                rows=[]
                for line in label.read_text(encoding="utf-8").splitlines():
                    p=line.split()
                    if len(p)>=5:
                        p[0]="0";rows.append(" ".join(p))
                if not rows: continue
                tag=hashlib.sha1(str(img.resolve()).encode()).hexdigest()[:10]
                fn=f"{n:07d}_{tag}{img.suffix.lower()}"
                shutil.copy2(img,args.out/outsplit/"images"/fn)
                (args.out/outsplit/"labels"/(Path(fn).stem+".txt")).write_text("\n".join(rows)+"\n",encoding="utf-8")
                manifest.append({"image":f"{outsplit}/images/{fn}","source":str(source),"source_split":split,"hash":h})
                n+=1
    (args.out/"data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n",encoding="utf-8")
    (args.out/"merge_manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest),encoding="utf-8")
    print(json.dumps({"kept":n,"duplicates_removed":dupes,"sources":[str(x) for x in args.sources]},indent=2))
if __name__=="__main__":main()
