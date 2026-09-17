from __future__ import annotations

import argparse, os, shutil, subprocess, sys, urllib.request
from pathlib import Path
import yaml

OPENOCR_URL='https://github.com/Topdu/OpenOCR.git'
PRETRAIN_URL='https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth'

def run(cmd,cwd=None):
    print('+',' '.join(map(str,cmd)),flush=True);subprocess.check_call(list(map(str,cmd)),cwd=cwd)

def ensure_repo(path:Path):
    if not (path/'.git').exists():
        path.parent.mkdir(parents=True,exist_ok=True);run(['git','clone','--depth','1',OPENOCR_URL,path])

def download(url:str,path:Path):
    if path.exists():return
    path.parent.mkdir(parents=True,exist_ok=True);print('Downloading',url);urllib.request.urlretrieve(url,path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('dataset',type=Path);ap.add_argument('--openocr',type=Path,default=Path('.vendor/OpenOCR'));ap.add_argument('--epochs',type=int,default=25);ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--output',type=Path,default=Path('gate_specialist_output'));args=ap.parse_args()
    ds=args.dataset.resolve();train=ds/'rec_gt_train.txt';val=ds/'rec_gt_test.txt'
    if not train.exists() or not val.exists():raise SystemExit('Run prepare_gate_dataset.py first.')
    ensure_repo(args.openocr);weights=(args.openocr/'openocr_svtrv2_ch.pth').resolve();download(PRETRAIN_URL,weights)
    base=args.openocr/'configs/rec/svtrv2/svtrv2_ch.yml';cfg=yaml.safe_load(base.read_text(encoding='utf-8'))
    cfg['Global']['pretrained_model']=str(weights);cfg['Global']['epoch_num']=args.epochs;cfg['Global']['save_model_dir']=str(args.output.resolve())
    for section,label in [('Train',train),('Eval',val)]:
        cfg[section]['dataset']['data_dir']=str(ds);cfg[section]['dataset']['label_file_list']=[str(label)]
        if section=='Train':cfg[section]['loader']['batch_size_per_card']=args.batch_size
    custom=args.output/'gate_svtrv2.yml';args.output.mkdir(parents=True,exist_ok=True);custom.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    env=os.environ.copy();env.setdefault('CUDA_VISIBLE_DEVICES','0')
    cmd=[sys.executable,'-m','torch.distributed.launch','--nproc_per_node=1','tools/train_rec.py','--c',str(custom.resolve())]
    print('Training gate-specialist SVTRv2 on confirmed/corrected site examples.')
    subprocess.check_call(cmd,cwd=args.openocr,env=env)
    best=next(args.output.rglob('best.pth'),None)
    if not best:
        print('Training finished; locate the best checkpoint under',args.output);return
    export_cfg=args.output/'gate_svtrv2_export.yml';shutil.copy2(custom,export_cfg)
    print('Best checkpoint:',best)
    print('To export ONNX:')
    print(sys.executable,'tools/toonnx.py --c',export_cfg.resolve(),'--o Global.device=cpu Global.pretrained_model='+str(best.resolve()))

if __name__=='__main__':main()
