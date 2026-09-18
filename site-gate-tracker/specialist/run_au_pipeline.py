#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(args, required=True):
    print("\n+", " ".join(map(str,args)), flush=True)
    p=subprocess.run(list(map(str,args)))
    if required and p.returncode:
        raise SystemExit(p.returncode)
    return p.returncode

def main():
    ap=argparse.ArgumentParser(description="Site Gate v7 Australian-specialist pipeline")
    ap.add_argument("--work",type=Path,default=Path("au_specialist_work"))
    ap.add_argument("--synthetic-count",type=int,default=250000)
    ap.add_argument("--roboflow",action="store_true",help="Download configured AU Roboflow sets; requires ROBOFLOW_API_KEY.")
    ap.add_argument("--plate-source",action="append",type=Path,default=[],help="Existing YOLO Australian plate dataset root; repeat as needed.")
    ap.add_argument("--gate-zip",type=Path,default=None,help="Exported v7 gate-specialist ZIP.")
    ap.add_argument("--train-detector",action="store_true")
    ap.add_argument("--train-ocr",action="store_true")
    ap.add_argument("--skip-official-data",action="store_true")
    args=ap.parse_args()
    py=sys.executable;w=args.work.resolve();w.mkdir(parents=True,exist_ok=True)

    if not args.skip_official_data:
        cmd=[py,HERE/"bootstrap_au_data.py","--out",w/"australia"]
        if args.roboflow:cmd.append("--roboflow")
        run(cmd)
        run([py,HERE/"build_au_priors.py",w/"australia","--out",w/"au_priors.json"],required=False)

    synth=w/"au_synth"
    if not (synth/"rec_gt_train.txt").exists():
        run([py,HERE/"generate_au_synthetic_plates.py","--count",str(args.synthetic_count),"--out",synth])

    sources=list(args.plate_source)
    if args.roboflow:
        for p in [(w/"australia/roboflow/nsw_white"),(w/"australia/roboflow/au_multistate")]:
            if p.exists():sources.append(p)
    merged=w/"au_plate_merged"
    if sources:
        run([py,HERE/"merge_au_plate_datasets.py",*sources,"--out",merged])
        if args.train_detector:
            run([py,HERE/"train_au_plate_detector.py",merged,"--project",w/"runs/plate"])
    elif args.train_detector:
        raise SystemExit("--train-detector requested but no plate dataset sources are available.")

    gate=None
    if args.gate_zip:
        gate=w/"gate_dataset"
        run([py,HERE/"prepare_gate_dataset.py",args.gate_zip,"--out",gate])

    if args.train_ocr:
        cmd=[py,HERE/"train_au_curriculum.py","--synthetic",synth,"--output",w/"runs/ocr"]
        if gate is not None:cmd += ["--gate",gate]
        run(cmd)

    print("\nAUSTRALIA SPECIALIST PIPELINE COMPLETE")
    print("workdir:",w)
    print("synthetic OCR:",synth)
    if sources: print("merged AU plate detector data:",merged)
    if gate: print("prepared physical-gate data:",gate)
    if not args.train_detector or not args.train_ocr:
        print("Training flags were not both enabled; data preparation is complete but requested model weights were not trained.")
    print("Production rule: research-only datasets are intentionally excluded from this pipeline.")

if __name__=="__main__":
    main()
