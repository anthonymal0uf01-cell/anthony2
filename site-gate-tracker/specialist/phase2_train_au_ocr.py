#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, math, random, string
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

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

def brier_records(records, calibrator=None):
    cal=calibrator or []
    if not records:return 0.0
    return sum(((apply_calibrator(r["raw_conf"],cal) if cal else r["raw_conf"])-r["ok"])**2 for r in records)/len(records)

def select_calibrator(records,bins=10):
    """Fit on one deterministic half of validation and select on the other.
    If empirical calibration does not improve held-out validation reliability,
    use identity calibration instead of damaging a good recogniser."""
    if len(records)<40:
        return [],{"mode":"identity","reason":"too_few_validation_records","check_n":len(records)}
    fit=records[::2]; check=records[1::2]
    candidate=fit_calibrator(fit,bins)
    raw=brier_records(check,[])
    calibrated=brier_records(check,candidate)
    if calibrated + 1e-6 < raw:
        return candidate,{"mode":"empirical_bins","fit_n":len(fit),"check_n":len(check),
                          "check_brier_raw":raw,"check_brier_calibrated":calibrated}
    return [],{"mode":"identity","reason":"empirical_bins_did_not_improve_holdout",
               "fit_n":len(fit),"check_n":len(check),
               "check_brier_raw":raw,"check_brier_calibrated":calibrated}

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
    ap.add_argument("--adapt-synth-ratio",type=float,default=1.5)
    ap.add_argument("--adapt-lr-mult",type=float,default=.06)
    ap.add_argument("--baseline",type=Path,default=None,
                    help="Currently deployed OCR checkpoint for hard-slice promotion comparison.")
    ap.add_argument("--init-checkpoint",type=Path,default=None,
                    help="Optional deployed OCR checkpoint to fine-tune instead of relearning from scratch.")
    ap.add_argument("--min-tiny-exact",type=float,default=.40)
    ap.add_argument("--min-tiny-improvement",type=float,default=.005)
    ap.add_argument("--specialist-mode",action="store_true",
                    help="Publish a separate tiny-plate specialist instead of replacing the general OCR.")
    ap.add_argument("--specialist-min-general-exact",type=float,default=.82)
    args=ap.parse_args()
    torch.manual_seed(1404);random.seed(1404);np.random.seed(1404)
    torch.set_num_threads(max(1,min(4,torch.get_num_threads())))
    tr=PlateDataset(args.dataset,args.dataset/"rec_gt_train.txt")
    va=PlateDataset(args.dataset,args.dataset/"rec_gt_val.txt")
    test_gt=args.dataset/"rec_gt_test.txt"
    te=PlateDataset(args.dataset,test_gt if test_gt.exists() else args.dataset/"rec_gt_val.txt")
    train=DataLoader(tr,batch_size=args.batch,shuffle=True,num_workers=2,collate_fn=collate,persistent_workers=True)
    val=DataLoader(va,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
    test=DataLoader(te,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinyAUOCR().to(device)
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        try:
            raw_init=torch.load(args.init_checkpoint,map_location="cpu",weights_only=False)
        except TypeError:
            raw_init=torch.load(args.init_checkpoint,map_location="cpu")
        init_state=raw_init.get("state_dict",raw_init) if isinstance(raw_init,dict) else raw_init
        model.load_state_dict(init_state)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.epochs),eta_min=args.lr*.08)
    ctc=nn.CTCLoss(blank=BLANK,zero_infinity=True)
    args.out.mkdir(parents=True,exist_ok=True)
    initial_metrics,_=eval_model(model,val,device)
    best=(initial_metrics["exact_match"],
          {k:v.detach().cpu() for k,v in model.state_dict().items()},
          dict(initial_metrics))
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
        sched.step()
        metrics,_=eval_model(model,val,device)
        metrics.update(epoch=epoch,train_loss=loss_sum/max(1,steps),lr=float(opt.param_groups[0]["lr"]))
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
        atest_gt=args.adapt_dataset/"rec_gt_test.txt"
        ate=PlateDataset(args.adapt_dataset,atest_gt if atest_gt.exists() else args.adapt_dataset/"rec_gt_val.txt")
        # Replay synthetic examples while adapting so Australian camera-domain tuning
        # cannot erase the general plate grammar learned in stage 1.
        rr=random.Random(260918)
        synth_n=min(len(tr),max(len(atr),int(len(atr)*args.adapt_synth_ratio)))
        synth_idx=rr.sample(range(len(tr)),synth_n)
        replay=Subset(tr,synth_idx)
        mixed=ConcatDataset([atr,atr,replay])
        adl=DataLoader(mixed,batch_size=args.batch,shuffle=True,num_workers=2,collate_fn=collate,persistent_workers=True)
        avl=DataLoader(ava,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
        atl=DataLoader(ate,batch_size=args.batch,shuffle=False,num_workers=2,collate_fn=collate,persistent_workers=True)
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr*args.adapt_lr_mult,weight_decay=1e-4)
        asched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,args.adapt_epochs),eta_min=args.lr*.01)
        best_adapt=(-1.0,None,None)
        for epoch in range(1,args.adapt_epochs+1):
            model.train();aloss=0.0;asteps=0
            for x,y,yl,_,_ in adl:
                x=x.to(device);y=y.to(device);yl=yl.to(device);logits=model(x)
                il=torch.full((x.shape[0],),logits.shape[1],dtype=torch.long,device=device)
                loss=ctc(logits.log_softmax(-1).transpose(0,1),y,il,yl)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step()
                aloss+=float(loss.detach());asteps+=1
            asched.step()
            am,_=eval_model(model,avl,device)
            sm,_=eval_model(model,val,device)
            # Harmonic mean punishes a checkpoint that improves one domain by destroying the other.
            comb=(2*am["exact_match"]*sm["exact_match"]/
                  max(1e-9,am["exact_match"]+sm["exact_match"]))
            print(json.dumps({"adapt_epoch":epoch,"camera_exact":am["exact_match"],
                "synthetic_exact":sm["exact_match"],"combined":comb,
                "train_loss":aloss/max(1,asteps),"lr":float(opt.param_groups[0]["lr"])}),flush=True)
            if comb>best_adapt[0]:
                ast={k:v.detach().cpu() for k,v in model.state_dict().items()}
                best_adapt=(comb,ast,{"camera_val":am,"synthetic_val":sm})
        if best_adapt[1] is not None:
            model.load_state_dict(best_adapt[1])
        adapt_metrics_final,_=eval_model(model,atl,device)

    # Compare the candidate against the currently deployed checkpoint on the exact
    # same camera-domain test set. This prevents a generic metric win from replacing
    # a model that is actually better on tiny street-camera plates.
    baseline_camera_metrics=None
    if args.baseline is not None and args.adapt_dataset is not None and args.baseline.exists():
        try:
            raw=torch.load(args.baseline,map_location="cpu",weights_only=False)
        except TypeError:
            raw=torch.load(args.baseline,map_location="cpu")
        state0=raw.get("state_dict",raw) if isinstance(raw,dict) else raw
        baseline_model=TinyAUOCR().to(device)
        baseline_model.load_state_dict(state0)
        baseline_camera_metrics,_=eval_model(baseline_model,atl,device)

    # Fit reliability bins on validation only, then report final metrics on untouched test.
    _,cal_records=eval_model(model,val,device)
    if args.adapt_dataset is not None:
        _,camera_cal_records=eval_model(model,avl,device)
        cal_records.extend(camera_cal_records)
    calibrator,calibration_selection=select_calibrator(cal_records,10)
    metrics,records=eval_model(model,test,device,calibrator)
    if args.adapt_dataset is not None:
        adapt_metrics_final,_=eval_model(model,atl,device,calibrator)
    metrics["calibrator"]=calibrator
    metrics["calibration_selection"]=calibration_selection
    metrics["training"]={
        "synthetic_train":len(tr),"synthetic_val":len(va),"synthetic_test":len(te),
        "epochs":args.epochs,"batch":args.batch,"lr":args.lr,
        "adapt_epochs":args.adapt_epochs if args.adapt_dataset is not None else 0,
        "adapt_synth_ratio":args.adapt_synth_ratio if args.adapt_dataset is not None else 0.0,
        "adapt_lr_mult":args.adapt_lr_mult if args.adapt_dataset is not None else 0.0,
        "camera_train":len(atr) if args.adapt_dataset is not None else 0,
        "camera_val":len(ava) if args.adapt_dataset is not None else 0,
        "camera_test":len(ate) if args.adapt_dataset is not None else 0,
    }
    hard=[v["exact_match"] for k,v in metrics["by_slice"].items() if k.startswith("cond:") and v["n"]>=25]
    metrics["worst_hard_slice"]=min(hard) if hard else metrics["exact_match"]
    if adapt_metrics_final is not None: metrics["camera_domain"]=adapt_metrics_final
    if baseline_camera_metrics is not None: metrics["baseline_camera_domain"]=baseline_camera_metrics
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    score=metrics["exact_match"]
    ckpt={"state_dict":state,"alphabet":ALPHABET,"input_width":160,"input_height":48,"metrics":metrics,
          "calibrator":calibrator,"model":"TinyAUOCR-v2-calibrated"}
    pt_name="au_ocr_tiny.pt" if args.specialist_mode else "au_ocr_seed.pt"
    onnx_name="au_ocr_tiny.onnx" if args.specialist_mode else "au_ocr_seed.onnx"
    metrics_name="au_ocr_tiny.metrics.json" if args.specialist_mode else "metrics.json"
    ckpt["specialist_mode"]=bool(args.specialist_mode)
    ckpt["specialist_target"]="tiny_plate_20_64" if args.specialist_mode else "general"
    torch.save(ckpt,args.out/pt_name)
    model.eval()
    # PyTorch checkpoint is the Phase-2 promotion artifact. ONNX is optional;
    # export incompatibilities must never discard a successfully trained model.
    try:
        dummy=torch.zeros(1,1,48,160)
        torch.onnx.export(model,dummy,args.out/onnx_name,input_names=["image"],output_names=["logits"],opset_version=17,dynamo=False)
    except Exception as e:
        print(f"ONNX_OPTIONAL_EXPORT_FAILED: {e}", flush=True)
    (args.out/metrics_name).write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (args.out/"README.txt").write_text(
        "Australian OCR seed trained from NSW/NHV synthetic curriculum.\n"
        f"validation exact_match={score:.4f} cer={metrics['cer']:.4f}\n"
        "This is a real phase-2 checkpoint; full SVTRv2 GPU fine-tuning remains the higher-capacity successor.\n",
        encoding="utf-8")
    if args.specialist_mode:
        if score < args.specialist_min_general_exact:
            raise SystemExit(
                f"specialist gate failed: retained general exact {score:.3f} < "
                f"{args.specialist_min_general_exact:.3f}")
    elif score < args.min_exact:
        raise SystemExit(f"promotion gate failed: exact_match {score:.3f} < {args.min_exact:.3f}")
    if metrics["worst_hard_slice"] < args.min_hard_slice:
        raise SystemExit(f"promotion gate failed: hard-slice exact {metrics['worst_hard_slice']:.3f} < {args.min_hard_slice:.3f}")
    if metrics["brier_calibrated"] > metrics["brier_raw"] + 0.002:
        raise SystemExit("promotion gate failed: selected validation-safe calibration materially worsened untouched-test reliability")
    if adapt_metrics_final is not None and adapt_metrics_final["exact_match"] < args.min_adapt_exact:
        raise SystemExit(f"promotion gate failed: camera-domain exact {adapt_metrics_final['exact_match']:.3f} < {args.min_adapt_exact:.3f}")
    if adapt_metrics_final is not None:
        tiny=adapt_metrics_final.get("by_slice",{}).get("cond:tiny_plate_20_64")
        if not tiny or tiny.get("n",0) < 25:
            raise SystemExit("promotion gate failed: insufficient tiny_plate_20_64 test examples")
        tiny_exact=float(tiny["exact_match"])
        metrics["tiny_plate_exact_match"]=tiny_exact
        if tiny_exact < args.min_tiny_exact:
            raise SystemExit(f"promotion gate failed: tiny-plate exact {tiny_exact:.3f} < {args.min_tiny_exact:.3f}")
        if baseline_camera_metrics is not None:
            bt=baseline_camera_metrics.get("by_slice",{}).get("cond:tiny_plate_20_64")
            if bt and bt.get("n",0)>=25:
                baseline_tiny=float(bt["exact_match"])
                metrics["baseline_tiny_plate_exact_match"]=baseline_tiny
                improvement=tiny_exact-baseline_tiny
                metrics["tiny_plate_improvement"]=improvement
                if improvement < args.min_tiny_improvement:
                    raise SystemExit(
                        f"promotion gate failed: tiny-plate improvement {improvement:.4f} "
                        f"< {args.min_tiny_improvement:.4f} (candidate={tiny_exact:.3f}, baseline={baseline_tiny:.3f})"
                    )
    # Rewrite metrics after all promotion-comparison fields have been added.
    (args.out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"PROMOTED exact_match={score:.4f} tiny_plate={metrics.get('tiny_plate_exact_match','n/a')}")

if __name__=="__main__":main()
