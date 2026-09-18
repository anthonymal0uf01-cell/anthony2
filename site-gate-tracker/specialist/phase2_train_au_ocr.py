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
        self.items=[]
        for line in gt.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            p,t=line.split("\t",1)
            t="".join(c for c in t.upper() if c in CHAR2ID)
            if t:self.items.append((p,t))
        self.width=width;self.height=height
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        p,t=self.items[i]
        im=Image.open(self.root/p).convert("L")
        im.thumbnail((self.width,self.height),Image.Resampling.LANCZOS)
        canvas=Image.new("L",(self.width,self.height),127)
        x=(self.width-im.width)//2;y=(self.height-im.height)//2
        canvas.paste(im,(x,y))
        a=np.asarray(canvas,dtype=np.float32)/255.0
        a=(a-.5)/.5
        x=torch.from_numpy(a)[None,:,:]
        ids=torch.tensor([CHAR2ID[c] for c in t],dtype=torch.long)
        return x,ids,t

def collate(batch):
    xs,ys,txt=zip(*batch)
    lens=torch.tensor([len(y) for y in ys],dtype=torch.long)
    return torch.stack(xs),torch.cat(ys),lens,list(txt)

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

def eval_model(model,loader,device):
    model.eval();n=exact=chars=edits=0
    def lev(a,b):
        d=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            nd=[i]
            for j,y in enumerate(b,1):nd.append(min(nd[-1]+1,d[j]+1,d[j-1]+(x!=y)))
            d=nd
        return d[-1]
    with torch.inference_mode():
        for x,_,_,truth in loader:
            pred=greedy(model(x.to(device)))
            for a,b in zip(truth,pred):
                n+=1;exact+=int(a==b);chars+=max(1,len(a));edits+=lev(a,b)
    return {"samples":n,"exact_match":exact/max(1,n),"cer":edits/max(1,chars)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("dataset",type=Path)
    ap.add_argument("--epochs",type=int,default=5)
    ap.add_argument("--batch",type=int,default=96)
    ap.add_argument("--lr",type=float,default=2e-3)
    ap.add_argument("--out",type=Path,default=Path("phase2_weights"))
    ap.add_argument("--min-exact",type=float,default=.72)
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
        for x,y,yl,_ in train:
            x=x.to(device);y=y.to(device);yl=yl.to(device)
            logits=model(x)
            T=logits.shape[1]
            il=torch.full((x.shape[0],),T,dtype=torch.long,device=device)
            loss=ctc(logits.log_softmax(-1).transpose(0,1),y,il,yl)
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
            loss_sum+=float(loss.detach());steps+=1
        metrics=eval_model(model,val,device)
        metrics.update(epoch=epoch,train_loss=loss_sum/max(1,steps))
        print(json.dumps(metrics),flush=True)
        if metrics["exact_match"]>best[0]:
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            best=(metrics["exact_match"],state,dict(metrics))
    score,state,metrics=best
    model.load_state_dict(state)
    ckpt={"state_dict":state,"alphabet":ALPHABET,"input_width":160,"input_height":48,"metrics":metrics,"model":"TinyAUOCR-v1"}
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
    print(f"PROMOTED exact_match={score:.4f}")

if __name__=="__main__":main()
