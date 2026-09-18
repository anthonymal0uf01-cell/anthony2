#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import yaml

def find_yaml(root:Path):
    cands=list(root.rglob("data.yaml"))
    if not cands: raise SystemExit(f"No data.yaml found under {root}")
    return cands[0]

def main():
    ap=argparse.ArgumentParser(description="Fine-tune an Australian plate detector on Roboflow YOLO exports.")
    ap.add_argument("dataset_roots",nargs="+",type=Path)
    ap.add_argument("--model",default="yolo26s.pt")
    ap.add_argument("--epochs",type=int,default=80)
    ap.add_argument("--imgsz",type=int,default=960)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--project",default="runs/au_plate")
    ap.add_argument("--name",default="au_plate_detector")
    args=ap.parse_args()
    if len(args.dataset_roots)!=1:
        raise SystemExit("Merge/dedupe datasets first; pass one canonical YOLO dataset root to avoid duplicate leakage.")
    data=find_yaml(args.dataset_roots[0])
    from ultralytics import YOLO
    model=YOLO(args.model)
    model.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,project=args.project,name=args.name,
                close_mosaic=10,cache=False,plots=True)
    model.val(data=str(data),imgsz=args.imgsz,plots=True)
    print("Australian plate detector training/evaluation complete.")

if __name__=="__main__":
    main()
