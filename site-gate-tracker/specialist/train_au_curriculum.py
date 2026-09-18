#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run(args):
    print("+"," ".join(map(str,args)),flush=True)
    subprocess.check_call(list(map(str,args)))

def best(root:Path):
    exact=list(root.rglob("best.pth"))
    if exact:return exact[0]
    anyp=list(root.rglob("*.pth"))
    return anyp[0] if anyp else None

def main():
    ap=argparse.ArgumentParser(description="Two-stage Australian SVTRv2 curriculum")
    ap.add_argument("--synthetic",type=Path,required=True,help="Output of generate_au_synthetic_plates.py")
    ap.add_argument("--gate",type=Path,default=None,help="Prepared corrected gate dataset")
    ap.add_argument("--output",type=Path,default=Path("au_specialist_output"))
    ap.add_argument("--openocr",type=Path,default=Path(".vendor/OpenOCR"))
    ap.add_argument("--synthetic-epochs",type=int,default=12)
    ap.add_argument("--gate-epochs",type=int,default=25)
    args=ap.parse_args()

    trainer=HERE/"train_gate_specialist.py"
    run([sys.executable,trainer,args.synthetic,"--openocr",args.openocr,"--epochs",str(args.synthetic_epochs),"--output",args.output,"--stage","01_au_synthetic"])
    ck=best(args.output/"01_au_synthetic")
    if not ck:
        raise SystemExit("Synthetic stage finished but no checkpoint was found.")

    if args.gate:
        run([sys.executable,trainer,args.gate,"--openocr",args.openocr,"--epochs",str(args.gate_epochs),"--output",args.output,"--stage","02_gate_specialist","--pretrained",ck])
        final=best(args.output/"02_gate_specialist")
        print("FINAL",final or "checkpoint not found")
    else:
        print("AU_BASE",ck)
        print("Gate stage skipped until corrected site examples are supplied.")

if __name__=="__main__":
    main()
