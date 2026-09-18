#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from PIL import Image

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--seed",type=int,default=260918);args=ap.parse_args()
    from huggingface_hub import snapshot_download
    root=Path(snapshot_download(repo_id="evan6007/TLPD",repo_type="dataset"))
    imgs=root/"images";labs=root/"labels"
    pairs=[]
    for ip in imgs.iterdir():
        if ip.suffix.lower() not in {".jpg",".jpeg",".png"}:continue
        jp=labs/(ip.stem+".json")
        if jp.exists():pairs.append((ip,jp))
    random.Random(args.seed).shuffle(pairs)
    n=len(pairs);cuts=(int(n*.80),int(n*.90));manifest=[]
    for i,(ip,jp) in enumerate(pairs):
        split="train" if i<cuts[0] else ("val" if i<cuts[1] else "test")
        od=args.out/split; (od/"images").mkdir(parents=True,exist_ok=True);(od/"labels").mkdir(parents=True,exist_ok=True)
        meta=json.loads(jp.read_text(encoding="utf-8",errors="replace"))
        W=float(meta.get("imageWidth") or Image.open(ip).width);H=float(meta.get("imageHeight") or Image.open(ip).height)
        rows=[]
        for s in meta.get("shapes",[]):
            pts=s.get("points") or []
            if len(pts)<2:continue
            xs=[float(p[0]) for p in pts];ys=[float(p[1]) for p in pts]
            x1,x2=max(0,min(xs)),min(W,max(xs));y1,y2=max(0,min(ys)),min(H,max(ys))
            bw=x2-x1;bh=y2-y1
            if bw<3 or bh<3:continue
            rows.append(f"0 {(x1+x2)/(2*W):.6f} {(y1+y2)/(2*H):.6f} {bw/W:.6f} {bh/H:.6f}")
        if not rows:continue
        fn=f"tlpd_{i:05d}{ip.suffix.lower()}";Image.open(ip).convert("RGB").save(od/"images"/fn,quality=92)
        (od/"labels"/(Path(fn).stem+".txt")).write_text("\n".join(rows)+"\n")
        manifest.append({"file":f"{split}/images/{fn}","source":"evan6007/TLPD","license":"MIT"})
    (args.out/"data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest))
    print(json.dumps({"samples":len(manifest),"source":"evan6007/TLPD","license":"MIT"},indent=2))
if __name__=="__main__":main()
