from __future__ import annotations

import argparse, os, subprocess, sys, urllib.request
from pathlib import Path
import yaml

OPENOCR_URL="https://github.com/Topdu/OpenOCR.git"
PRETRAIN_URL="https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth"

def run(cmd,cwd=None,env=None):
    print("+"," ".join(map(str,cmd)),flush=True)
    subprocess.check_call(list(map(str,cmd)),cwd=cwd,env=env)

def ensure_repo(path:Path):
    if not (path/".git").exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        run(["git","clone","--depth","1",OPENOCR_URL,path])

def download(url:str,path:Path):
    if path.exists(): return
    path.parent.mkdir(parents=True,exist_ok=True)
    print("Downloading",url)
    urllib.request.urlretrieve(url,path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset",type=Path)
    ap.add_argument("--openocr",type=Path,default=Path(".vendor/OpenOCR"))
    ap.add_argument("--epochs",type=int,default=25)
    ap.add_argument("--batch-size",type=int,default=64)
    ap.add_argument("--output",type=Path,default=Path("gate_specialist_output"))
    ap.add_argument("--pretrained",type=Path,default=None,help="Weights for transfer learning. Defaults to official SVTRv2 server weights.")
    ap.add_argument("--checkpoint",type=Path,default=None,help="Resume an interrupted OpenOCR stage.")
    ap.add_argument("--stage",default="gate")
    args=ap.parse_args()

    ds=args.dataset.resolve();train=ds/"rec_gt_train.txt";val=ds/"rec_gt_val.txt"
    if not val.exists(): val=ds/"rec_gt_test.txt"
    if not train.exists() or not val.exists():
        raise SystemExit("Dataset must contain rec_gt_train.txt and rec_gt_val.txt or rec_gt_test.txt.")

    ensure_repo(args.openocr)
    official=(args.openocr/"openocr_svtrv2_ch.pth").resolve()
    if args.pretrained is None and args.checkpoint is None:
        download(PRETRAIN_URL,official)
    base=args.openocr/"configs/rec/svtrv2/svtrv2_ch.yml"
    cfg=yaml.safe_load(base.read_text(encoding="utf-8"))
    cfg["Global"]["epoch_num"]=args.epochs
    cfg["Global"]["save_model_dir"]=str((args.output/args.stage).resolve())
    if args.checkpoint:
        cfg["Global"]["checkpoints"]=str(args.checkpoint.resolve())
    else:
        cfg["Global"]["pretrained_model"]=str((args.pretrained or official).resolve())

    for section,label in [("Train",train),("Eval",val)]:
        cfg[section]["dataset"]["data_dir"]=str(ds)
        cfg[section]["dataset"]["label_file_list"]=[str(label)]
        if section=="Train":
            cfg[section]["loader"]["batch_size_per_card"]=args.batch_size

    stage_dir=args.output/args.stage
    stage_dir.mkdir(parents=True,exist_ok=True)
    custom=stage_dir/f"{args.stage}_svtrv2.yml"
    custom.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
    env=os.environ.copy();env.setdefault("CUDA_VISIBLE_DEVICES","0")
    run([sys.executable,"-m","torch.distributed.launch","--nproc_per_node=1","tools/train_rec.py","--c",str(custom.resolve())],cwd=args.openocr,env=env)
    candidates=sorted(stage_dir.rglob("best.pth"))+sorted(stage_dir.rglob("*.pth"))
    if candidates:
        print("CHECKPOINT",candidates[0].resolve())
    else:
        print("Training command completed; inspect",stage_dir,"for OpenOCR checkpoints.")

if __name__=="__main__":
    main()
