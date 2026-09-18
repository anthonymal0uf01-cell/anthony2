#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hf",type=Path,required=True)
    ap.add_argument("--au",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--model",default="yolo26n.pt")
    args=ap.parse_args()
    from ultralytics import YOLO
    args.out.mkdir(parents=True,exist_ok=True)
    m=YOLO(args.model)
    # Stage 1: generic CC-BY localisation warm-up.
    r1=m.train(data=str(args.hf/"data.yaml"),epochs=2,imgsz=416,batch=16,workers=2,project=str(args.out),name="01_hf_warmup",plots=False,cache=False,verbose=False)
    p1=Path(r1.save_dir)/"weights"/"best.pt"
    m=YOLO(str(p1))
    # Stage 2: Australian road-domain adaptation.
    r2=m.train(data=str(args.au/"data.yaml"),epochs=8,imgsz=512,batch=12,workers=2,project=str(args.out),name="02_au_domain",plots=False,cache=False,close_mosaic=2,verbose=False)
    best=Path(r2.save_dir)/"weights"/"best.pt"
    v=m.val(data=str(args.au/"data.yaml"),imgsz=512,plots=False,verbose=False)
    metrics={
      "map50":float(v.box.map50),"map50_95":float(v.box.map),
      "precision":float(v.box.mp),"recall":float(v.box.mr),
      "hf_source":"justjuu/license-plate-detection CC BY 4.0",
      "au_source":"TfNSW Live Traffic Cameras CC BY + synthetic NSW/NHV overlays",
      "model":args.model
    }
    shutil.copy2(best,args.out/"au_plate_detector.pt")
    (args.out/"au_plate_detector.metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    if metrics["map50"]<0.55 or metrics["recall"]<0.55:
        raise SystemExit("Detector promotion gate failed")
if __name__=="__main__":main()
