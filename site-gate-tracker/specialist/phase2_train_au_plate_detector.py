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
    ap.add_argument("--baseline",type=Path,default=None,
                    help="Currently deployed detector checkpoint for tiny-plate comparison.")
    ap.add_argument("--tiny-max-width",type=int,default=64)
    ap.add_argument("--min-tiny-recall",type=float,default=.35)
    ap.add_argument("--min-tiny-improvement",type=float,default=.005)
    args=ap.parse_args()
    from ultralytics import YOLO
    args.out.mkdir(parents=True,exist_ok=True)
    m=YOLO(args.model)
    # Optional generic warm-up. For focused improvement of an already-deployed AU
    # checkpoint, pass --warmup-epochs 0 and --model <current checkpoint>.
    if args.warmup_epochs > 0:
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

    # Build a deterministic test subset containing the tiny 20..N px plates that caused
    # the live TfNSW failure. Candidate and deployed baseline are evaluated identically.
    tiny_root=args.out/"tiny_eval"
    tiny_img=tiny_root/"images"; tiny_lab=tiny_root/"labels"
    tiny_img.mkdir(parents=True,exist_ok=True); tiny_lab.mkdir(parents=True,exist_ok=True)
    tiny_n=0
    mf=args.au/"manifest.jsonl"
    if mf.exists():
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            try: rec=json.loads(line)
            except Exception: continue
            rel=Path(rec.get("file",""))
            if len(rel.parts)<3 or rel.parts[0]!="test": continue
            objs=rec.get("objects") or []
            widths=[float(o.get("bbox",[0,0,0,0])[2]) for o in objs if o.get("bbox")]
            if not widths or not any(20 <= w <= args.tiny_max_width for w in widths): continue
            src=args.au/rel
            lab=args.au/"test"/"labels"/(src.stem+".txt")
            if src.exists() and lab.exists():
                shutil.copy2(src,tiny_img/src.name); shutil.copy2(lab,tiny_lab/lab.name); tiny_n+=1
    tiny_yaml=tiny_root/"data.yaml"
    tiny_yaml.write_text(
        f"path: {tiny_root.resolve()}\ntrain: images\nval: images\ntest: images\nnames:\n  0: license_plate\n",
        encoding="utf-8")
    tiny_candidate=None; tiny_baseline=None
    if tiny_n>=25:
        tv=mb.val(data=str(tiny_yaml),split="val",imgsz=args.au_imgsz,plots=False,verbose=False)
        tiny_candidate={"n_images":tiny_n,"map50":float(tv.box.map50),"recall":float(tv.box.mr),
                        "precision":float(tv.box.mp)}
        if args.baseline is not None and args.baseline.exists():
            bm=YOLO(str(args.baseline))
            bv=bm.val(data=str(tiny_yaml),split="val",imgsz=args.au_imgsz,plots=False,verbose=False)
            tiny_baseline={"n_images":tiny_n,"map50":float(bv.box.map50),"recall":float(bv.box.mr),
                           "precision":float(bv.box.mp)}

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
      "tiny_plate_eval":tiny_candidate,
      "baseline_tiny_plate_eval":tiny_baseline,
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
    if tiny_candidate is None:
        raise SystemExit("Detector promotion gate failed: insufficient tiny-plate test examples")
    if tiny_candidate["recall"] < args.min_tiny_recall:
        raise SystemExit(
            f"Detector promotion gate failed: tiny recall={tiny_candidate['recall']:.3f} < {args.min_tiny_recall:.3f}")
    if tiny_baseline is not None:
        improvement=tiny_candidate["recall"]-tiny_baseline["recall"]
        metrics["tiny_recall_improvement"]=improvement
        (args.out/"au_plate_detector.metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        if improvement < args.min_tiny_improvement:
            raise SystemExit(
                f"Detector promotion gate failed: tiny recall improvement={improvement:.4f} "
                f"< {args.min_tiny_improvement:.4f} "
                f"(candidate={tiny_candidate['recall']:.3f}, baseline={tiny_baseline['recall']:.3f})")
if __name__=="__main__":main()
