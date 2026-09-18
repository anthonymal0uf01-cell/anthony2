from __future__ import annotations

import argparse, json, random, re, shutil, zipfile
from pathlib import Path

import cv2
import numpy as np
from tools.infer_e2e import OpenOCRE2E

ALNUM = re.compile(r"[^A-Z0-9]")

def norm(s: str) -> str:
    return ALNUM.sub("", (s or "").upper())[:10]

def edit_distance(a: str, b: str) -> int:
    a,b=norm(a),norm(b)
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        nd=[i]
        for j,cb in enumerate(b,1):
            nd.append(min(nd[-1]+1,dp[j]+1,dp[j-1]+(ca!=cb)))
        dp=nd
    return dp[-1]

def order_quad(points):
    pts=np.asarray(points,dtype=np.float32).reshape(4,2);s=pts.sum(1);d=np.diff(pts,axis=1).ravel()
    return np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],np.float32)

def rectify(img,points):
    q=order_quad(points);tl,tr,br,bl=q
    w=max(16,int(max(np.linalg.norm(tr-tl),np.linalg.norm(br-bl))));h=max(8,int(max(np.linalg.norm(bl-tl),np.linalg.norm(br-tr))))
    dst=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(q,dst),(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def box_score(item,label,shape):
    txt=norm(item.get('transcription',''));score=float(item.get('score',0) or 0);pts=np.asarray(item.get('points',[]),np.float32)
    if pts.size<8:return -999
    x0,y0=pts[:,0].min(),pts[:,1].min();x1,y1=pts[:,0].max(),pts[:,1].max();w=max(1,x1-x0);h=max(1,y1-y0);ratio=w/h
    sim=1-edit_distance(txt,label)/max(1,len(label)) if txt else 0
    plate_shape=0.25 if .75<=ratio<=8 else -0.15
    size=min(.15,(w*h/max(1,shape[0]*shape[1]))**.5)
    return score+.65*sim+plate_shape+size

def main():
    ap=argparse.ArgumentParser();ap.add_argument('zip',type=Path);ap.add_argument('--out',type=Path,default=Path('gate_dataset'));ap.add_argument('--val-frac',type=float,default=.2);args=ap.parse_args()
    work=args.out/'_export';args.out.mkdir(parents=True,exist_ok=True)
    if work.exists():shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(args.zip) as z:z.extractall(work)
    manifest=[json.loads(x) for x in (work/'manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    labelled=[m for m in manifest if norm(m.get('label','')) or norm(m.get('label2',''))]
    if not labelled:raise SystemExit('No labelled samples in export. Confirm plates or use LABEL LATEST HARD CASE first.')
    ocr=OpenOCRE2E(mode='mobile',backend='onnx',drop_score=.05,det_box_type='quad',use_gpu='auto')
    rows=[]
    crops=args.out/'crops';crops.mkdir(exist_ok=True)
    for n,m in enumerate(labelled):
        src=work/m['image'];img=cv2.imread(str(src))
        if img is None:continue
        result,_=ocr(img_numpy_list=[img],is_visualize=False,crop_infer=True,rec_batch_num=4);items=(result or [[]])[0]
        if not items:continue
        used=set()
        labels=[("primary",norm(m.get("label",""))),("secondary",norm(m.get("label2","")))]
        for slot,label in labels:
            if not label:continue
            ranked=sorted(enumerate(items),key=lambda kv:box_score(kv[1],label,img.shape),reverse=True)
            chosen=None
            for idx,it in ranked:
                if idx in used:continue
                if box_score(it,label,img.shape)>=.15:
                    chosen=(idx,it);break
            if chosen is None:continue
            idx,best=chosen;used.add(idx)
            crop=rectify(img,best['points'])
            if crop.shape[1]<20 or crop.shape[0]<8:continue
            fn=f"{n:06d}_{slot}_{label}.jpg";cv2.imwrite(str(crops/fn),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
            meta=dict(m);meta["plate_slot"]=slot
            rows.append((f'crops/{fn}',label,meta))
    if len(rows)<4:raise SystemExit(f'Only {len(rows)} usable labelled plate crops found. Collect/correct more hard cases.')
    random.Random(1404).shuffle(rows);cut=max(1,int(len(rows)*(1-args.val_frac)));train,val=rows[:cut],rows[cut:]
    (args.out/'rec_gt_train.txt').write_text(''.join(f'{p}\t{y}\n' for p,y,_ in train),encoding='utf-8')
    (args.out/'rec_gt_test.txt').write_text(''.join(f'{p}\t{y}\n' for p,y,_ in val),encoding='utf-8')
    summary={'source_samples':len(labelled),'usable_crops':len(rows),'train':len(train),'val':len(val),'conditions':{}}
    for _,_,m in rows:
        q=m.get('quality') or {};tags=[]
        if q.get('dark',0)>.12:tags.append('night/dark')
        if q.get('glare',0)>.05:tags.append('headlight/glare')
        if q.get('sharpness',1)<.35:tags.append('motion/blur')
        for t in tags:summary['conditions'][t]=summary['conditions'].get(t,0)+1
    (args.out/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
