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
    ap.add_argument("--warmup-epochs",type=int,default=1)
    ap.add_argument("--au-epochs",type=int,default=5)
    ap.add_argument("--warmup-imgsz",type=int,default=320)
    ap.add_argument("--au-imgsz",type=int,default=384)
    ap.add_argument("--warmup-batch",type=int,default=32)
    ap.add_argument("--au-batch",type=int,default=24)
    ap.add_argument("--min-map50",type=float,default=.60)
    ap.add_argument("--min-recall",type=float,default=.60)
    args=ap.parse_args()
    from ultralytics import YOLO
    args.out.mkdir(parents=True,exist_ok=True)
    m=YOLO(args.model)
    # Stage 1: generic CC-BY localisation warm-up.
    r1=m.train(data=str(args.hf/"data.yaml"),epochs=args.warmup_epochs,imgsz=args.warmup_imgsz,batch=args.warmup_batch,workers=2,
        project=str(args.out),name="01_generic_warmup",plots=False,cache=False,verbose=False,
        patience=max(2,args.warmup_epochs),cos_lr=True,close_mosaic=1)
    p1=Path(r1.save_dir)/"weights"/"best.pt"
    m=YOLO(str(p1))
    # Stage 2: Australian road-domain adaptation.
    r2=m.train(data=str(args.au/"data.yaml"),epochs=args.au_epochs,imgsz=args.au_imgsz,batch=args.au_batch,workers=2,
        project=str(args.out),name="02_au_domain",plots=False,cache=False,close_mosaic=3,
        patience=max(4,args.au_epochs//3),cos_lr=True,verbose=False)
    best=Path(r2.save_dir)/"weights"/"best.pt"
    mb=YOLO(str(best))
    # Promotion is measured on the untouched Australian test split.
    v=mb.val(data=str(args.au/"data.yaml"),split="test",imgsz=args.au_imgsz,plots=False,verbose=False)
    def count_images(root:Path,split:str)->int:
        d=root/split/"images"
        return sum(1 for x in d.glob("*") if x.suffix.lower() in {".jpg",".jpeg",".png",".webp"}) if d.exists() else 0
    metrics={
      "map50":float(v.box.map50),"map50_95":float(v.box.map),
      "precision":float(v.box.mp),"recall":float(v.box.mr),
      "hf_source":"justjuu/license-plate-detection CC BY 4.0",
      "au_source":"TfNSW Live Traffic Cameras CC BY + synthetic NSW/NHV overlays",
      "model":args.model,"warmup_epochs":args.warmup_epochs,"au_epochs":args.au_epochs,
      "warmup_imgsz":args.warmup_imgsz,"au_imgsz":args.au_imgsz,
      "warmup_batch":args.warmup_batch,"au_batch":args.au_batch,
      "evaluation_split":"test",
      "dataset_counts":{
        "generic_train":count_images(args.hf,"train"),
        "generic_val":count_images(args.hf,"val"),
        "generic_test":count_images(args.hf,"test"),
        "au_train":count_images(args.au,"train"),
        "au_val":count_images(args.au,"val"),
        "au_test":count_images(args.au,"test")
      }
    }
    shutil.copy2(best,args.out/"au_plate_detector.pt")
    try:
        exported=Path(mb.export(format="onnx",imgsz=args.au_imgsz,opset=17,simplify=False,dynamic=False))
        if exported.exists(): shutil.copy2(exported,args.out/"au_plate_detector.onnx")
    except Exception as e:
        print("DETECTOR_ONNX_EXPORT_FAILED",repr(e),flush=True)
    (args.out/"au_plate_detector.metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2))
    if metrics["map50"]<args.min_map50 or metrics["recall"]<args.min_recall:
        raise SystemExit(f"Detector promotion gate failed: map50={metrics['map50']:.3f}, recall={metrics['recall']:.3f}")
if __name__=="__main__":main()
