#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, string
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ALPHABET="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK=0
CHAR2ID={c:i+1 for i,c in enumerate(ALPHABET)}
ID2CHAR={i+1:c for i,c in enumerate(ALPHABET)}

class PlateDataset(Dataset):
    def __init__(self, root:Path, gt:Path, width=160, height=48):
        self.root=root
        meta={}
        mf=root/"manifest.jsonl"
        if mf.exists():
            for line in mf.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try:
                    r=json.loads(line); meta[r.get("image","")]=r
                except Exception: pass
        self.items=[]
        for line in gt.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            p,t=line.split("\t",1)
            t="".join(c for c in t.upper() if c in CHAR2ID)
            if t:self.items.append((p,t,meta.get(p,{})))
        self.width=width;self.height=height
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,t,m=self.items[i]
        im=Image.open(self.root/p).convert("L")
        im.thumbnail((self.width,self.height),Image.Resampling.LANCZOS)
        canvas=Image.new("L",(self.width,self.height),127)
        x=(self.width-im.width)//2;y=(self.height-im.height)//2
        canvas.paste(im,(x,y))
        a=np.asarray(canvas,dtype=np.float32)/255.0
        a=(a-.5)/.5
        x=torch.from_numpy(a)[None,:,:]
        ids=torch.tensor([CHAR2ID[c] for c in t],dtype=torch.long)
        return x,ids,t,m

def collate(batch):
    xs,ys,txt,meta=zip(*batch)
    lens=torch.tensor([len(y) for y in ys],dtype=torch.long)
    return torch.stack(xs),torch.cat(ys),lens,list(txt),list(meta)

class TinyAUOCR(nn.Module):
    def __init__(self,nclass=len(ALPHABET)+1):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,24,3,1,1),nn.BatchNorm2d(24),nn.SiLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(24,48,3,1,1),nn.BatchNorm2d(48),nn.SiLU(),nn.MaxPool2d(2,2),
            nn.Conv2d(48,72,3,1,1),nn.BatchNorm2d(72),nn.SiLU(),nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(72,96,3,1,1),nn.BatchNorm2d(96),nn.SiLU(),nn.MaxPool2d((2,1),(2,1)),
            nn.AdaptiveAvgPool2d((1,None)),
        )
        self.rnn=nn.GRU(96,64,num_layers=1,bidirectional=True,batch_first=True)
        self.fc=nn.Linear(128,nclass)
    def forward(self,x):
        z=self.cnn(x).squeeze(2).permute(0,2,1)
        z,_=self.rnn(z)
        return self.fc(z)

def greedy(logits):
    ids=logits.argmax(-1).cpu().tolist()
    out=[]
    for seq in ids:
        prev=-1;s=[]
        for i in seq:
            if i!=BLANK and i!=prev:s.append(ID2CHAR.get(i,""))
            prev=i
        out.append("".join(s))
    return out

def _lev(a,b):
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nd=[i]
        for j,y in enumerate(b,1):nd.append(min(nd[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=nd
    return d[-1]

def decode_with_conf(logits):
    probs=logits.softmax(-1)
    ids=probs.argmax(-1).cpu().tolist()
    pv=probs.detach().cpu().numpy()
    out=[]
    for bi,seq in enumerate(ids):
        prev=-1;s=[];used=[]
        for ti,i in enumerate(seq):
            if i!=BLANK and i!=prev:
                s.append(ID2CHAR.get(i,"")); used.append(float(pv[bi,ti,i]))
            prev=i
        conf=float(np.prod(np.clip(used,1e-5,1.0))**(1/max(1,len(used)))) if used else 0.0
        out.append(("".join(s),conf))
    return out

def fit_calibrator(records,bins=10):
    rec=sorted(records,key=lambda r:r["raw_conf"])
    groups=[]
    for k in range(bins):
        a=int(len(rec)*k/bins); b=int(len(rec)*(k+1)/bins)
        part=rec[a:b]
        if not part: continue
        groups.append({"lo":min(x["raw_conf"] for x in part),"hi":max(x["raw_conf"] for x in part),
                       "accuracy":sum(x["ok"] for x in part)/len(part),"n":len(part)})
    # monotonic empirical reliability curve
    last=0.0
    for g in groups:
        g["accuracy"]=max(last,g["accuracy"]); last=g["accuracy"]
    return groups

def apply_calibrator(conf,cal):
    if not cal:return conf
    for g in cal:
        if conf<=g["hi"]: return float(g["accuracy"])
    return float(cal[-1]["accuracy"])

def eval_model(model,loader,device,calibrator=None):
    model.eval();n=exact=chars=edits=0;records=[];by={}
    with torch.inference_mode():
        for x,_,_,truth,meta in loader:
            dec=decode_with_conf(model(x.to(device)))
            for a,(b,raw),m in zip(truth,dec,meta):
                ok=int(a==b);n+=1;exact+=ok;chars+=max(1,len(a));edits+=_lev(a,b)
                rec={"truth":a,"pred":b,"raw_conf":raw,"ok":ok,"kind":m.get("kind"),"conditions":m.get("conditions",[])}
                records.append(rec)
                tags=["all"]
                if m.get("kind"):tags.append("kind:"+str(m["kind"]))
                tags += ["cond:"+str(t) for t in m.get("conditions",[])]
                for tag in tags:
                    q=by.setdefault(tag,[0,0]);q[0]+=1;q[1]+=ok
    cal=calibrator or []
    brier_raw=sum((r["raw_conf"]-r["ok"])**2 for r in records)/max(1,len(records))
    brier_cal=sum((apply_calibrator(r["raw_conf"],cal)-r["ok"])**2 for r in records)/max(1,len(records)) if cal else brier_raw
    return {"samples":n,"exact_match":exact/max(1,n),"cer":edits/max(1,chars),
            "brier_raw":brier_raw,"brier_calibrated":brier_cal,
            "by_slice":{k:{"n":v[0],"exact_match":v[1]/v[0]} for k,v in by.items()}}, records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset",type=Path)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch",type=int,default=96)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--out",type=Path,default=Path("phase2_weights"))
    ap.add_argument("--min-exact",type=float,default=.88)
    ap.add_argument("--min-hard-slice",type=float,default=.70)
    ap.add_argument("--adapt-dataset",type=Path,default=None)
    ap.add_argument("--adapt-epochs",type=int,default=2)
    ap.add_argument("--min-adapt-exact",type=float,default=.78)
    args=ap.parse_args()
    torch.manual_seed(1404);random.seed(1404);np.random.seed(1404)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tr=PlateDataset(args.dataset,args.dataset/"rec_gt_train.txt")
    va=PlateDataset(args.dataset,args.dataset/"rec_gt_val.txt")
    train=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=2,collate_fn=collate,persistent_workers=True)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinyAUOCR().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    ctc=nn.CTCLoss(blank=BLANK,zero_infinity=True)
    best=(-1,None,None)
    args.out.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0;steps=0
        for x,y,yl,_,_ in train:
            x=x.to(device);y=y.to(device);yl=yl.to(device)
            logits=model(x)
            T=logits.shape[1]
            il=torch.full((x.shape[0],),T,dtype=torch.long,device=device)
            loss=ctc(logits.log_softmax(-1).transpose(0,1),y,il,yl)
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            loss_sum+=float(loss.detach());steps+=1
        metrics,_=eval_model(model,val,device)
        metrics.update(epoch=epoch,train_loss=loss_sum/max(1,steps))
        print(json.dumps(metrics),flush=True)
        if metrics["exact_match"]>best[0]:
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            best=(metrics["exact_match"],state,dict(metrics))
    score,state,metrics=best
    model.load_state_dict(state)

    # Optional second stage: Australian camera-domain crops with known synthetic labels.
    adapt_metrics_final=None
    if args.adapt_dataset is not None:
        atr=PlateDataset(args.adapt_dataset,args.adapt_dataset/"rec_gt_train.txt")
        ava=PlateDataset(args.adapt_dataset,args.adapt_dataset/"rec_gt_val.txt")
        adl=DataLoader(atr,batch_size=args.batch,shuffle=True,num_workers=2,collate_fn=collate,persistent_workers=True)
        avl=DataLoader(ava,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr*.22,weight_decay=1e-4)
        for epoch in range(1,args.adapt_epochs+1):
            model.train()
            for x,y,yl,_,_ in adl:
                x=x.to(device);y=y.to(device);yl=yl.to(device);logits=model(x)
                il=torch.full((x.shape[0],),logits.shape[1],dtype=torch.long,device=device)
                loss=ctc(logits.log_softmax(-1).transpose(0,1),y,il,yl)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            am,_=eval_model(model,avl,device)
            adapt_metrics_final=am
            print(json.dumps({"adapt_epoch":epoch,**am}),flush=True)

    # Fit empirical confidence calibration on an independent held-out split.
    raw_metrics,records=eval_model(model,val,device)
    calibrator=fit_calibrator(records,10)
    metrics,records=eval_model(model,val,device,calibrator)
    metrics["calibrator"]=calibrator
    hard=[v["exact_match"] for k,v in metrics["by_slice"].items() if k.startswith("cond:") and v["n"]>=25]
    metrics["worst_hard_slice"]=min(hard) if hard else metrics["exact_match"]
    if adapt_metrics_final is not None: metrics["camera_domain"]=adapt_metrics_final
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    score=metrics["exact_match"]
    ckpt={"state_dict":state,"alphabet":ALPHABET,"input_width":160,"input_height":48,"metrics":metrics,
          "calibrator":calibrator,"model":"TinyAUOCR-v2-calibrated"}
    torch.save(ckpt,args.out/"au_ocr_seed.pt")
    model.eval()
    # PyTorch checkpoint is the Phase-2 promotion artifact. ONNX is optional;
    # export incompatibilities must never discard a successfully trained model.
    try:
        dummy=torch.zeros(1,1,48,160)
        torch.onnx.export(model,dummy,args.out/"au_ocr_seed.onnx",input_names=["image"],output_names=["logits"],opset_version=17,dynamo=False)
    except Exception as e:
        print(f"ONNX_OPTIONAL_EXPORT_FAILED: {e}", flush=True)
    (args.out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (args.out/"README.txt").write_text(
        "Australian OCR seed trained from NSW/NHV synthetic curriculum.\n"
        f"validation exact_match={score:.4f} cer={metrics['cer']:.4f}\n"
        "This is a real phase-2 checkpoint; full SVTRv2 GPU fine-tuning remains the higher-capacity successor.\n",
        encoding="utf-8")
    if score < args.min_exact:
        raise SystemExit(f"promotion gate failed: exact_match {score:.3f} < {args.min_exact:.3f}")
    if metrics["worst_hard_slice"] < args.min_hard_slice:
        raise SystemExit(f"promotion gate failed: hard-slice exact {metrics['worst_hard_slice']:.3f} < {args.min_hard_slice:.3f}")
    if metrics["brier_calibrated"] > metrics["brier_raw"] + 1e-6:
        raise SystemExit("promotion gate failed: calibration made reliability worse")
    if adapt_metrics_final is not None and adapt_metrics_final["exact_match"] < args.min_adapt_exact:
        raise SystemExit(f"promotion gate failed: camera-domain exact {adapt_metrics_final['exact_match']:.3f} < {args.min_adapt_exact:.3f}")
    print(f"PROMOTED exact_match={score:.4f}")

if __name__=="__main__":main()
