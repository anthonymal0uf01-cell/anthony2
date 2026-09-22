#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw
import numpy as np

def ultralytics_model(repo_id, filename):
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    p=hf_hub_download(repo_id=repo_id,filename=filename)
    return YOLO(p)

def run_ultra(model, images, imgsz=768, conf=.05):
    out={}
    for p in images:
        im=Image.open(p).convert("RGB")
        r=model.predict(source=np.asarray(im),imgsz=imgsz,conf=conf,iou=.5,verbose=False)[0]
        rows=[]
        if r.boxes is not None:
            for b,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()):
                x1,y1,x2,y2=[float(x) for x in b]
                w=x2-x1; h=y2-y1; ar=w/max(1,h)
                if w*h<20 or ar<1.0 or ar>10: continue
                rows.append({"box":[x1,y1,x2,y2],"conf":float(c)})
        out[str(p)]=rows
    return out

def run_yolos(repo_id, images, threshold=.05):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    proc=AutoImageProcessor.from_pretrained(repo_id)
    model=AutoModelForObjectDetection.from_pretrained(repo_id).eval()
    out={}
    for p in images:
        im=Image.open(p).convert("RGB")
        x=proc(images=im,return_tensors="pt")
        with torch.no_grad(): y=model(**x)
        target=torch.tensor([im.size[::-1]])
        res=proc.post_process_object_detection(y,threshold=threshold,target_sizes=target)[0]
        rows=[]
        for b,c in zip(res["boxes"],res["scores"]):
            x1,y1,x2,y2=[float(v) for v in b.tolist()]
            w=x2-x1; h=y2-y1; ar=w/max(1,h)
            if w*h<20 or ar<1.0 or ar>10: continue
            rows.append({"box":[x1,y1,x2,y2],"conf":float(c)})
        out[str(p)]=rows
    return out

def stats(rows):
    n=len(rows); with_box=sum(bool(v) for v in rows.values())
    boxes=sum(len(v) for v in rows.values())
    confs=[x["conf"] for v in rows.values() for x in v]
    return {
      "images":n,
      "images_with_box":with_box,
      "coverage":with_box/max(1,n),
      "boxes":boxes,
      "mean_conf":sum(confs)/len(confs) if confs else 0,
      "max_conf":max(confs) if confs else 0,
    }

def overlay_grid(images, results_by_model, out_path):
    names=list(results_by_model)
    thumbw,thumbh=220,160
    rows=[]
    for p in images:
        base=Image.open(p).convert("RGB")
        ratio=min(thumbw/base.width,thumbh/base.height)
        nw,nh=max(1,int(base.width*ratio)),max(1,int(base.height*ratio))
        for name in names:
            im=base.copy(); d=ImageDraw.Draw(im)
            for z in results_by_model[name].get(str(p),[]):
                b=z["box"]; d.rectangle(tuple(b),outline=(255,0,0),width=max(1,int(2/ratio)))
                d.text((b[0],max(0,b[1]-10)),f'{z["conf"]:.2f}',fill=(255,0,0))
            im.thumbnail((thumbw,thumbh))
            tile=Image.new("RGB",(thumbw,thumbh+22),"white")
            tile.paste(im,((thumbw-im.width)//2,0))
            td=ImageDraw.Draw(tile);td.text((3,thumbh+3),name,fill="black")
            rows.append((p,name,tile))
    cols=len(names)
    grid=Image.new("RGB",(cols*thumbw,len(images)*(thumbh+22)),"white")
    for i,p in enumerate(images):
        for j,name in enumerate(names):
            tile=rows[i*cols+j][2]
            grid.paste(tile,(j*thumbw,i*(thumbh+22)))
    grid.save(out_path,quality=90)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--crops",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=60)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    images=sorted(a.crops.rglob("*.jpg"))[:a.limit]
    if not images: raise SystemExit("No crops found")

    models={}
    ours=Path("site-gate-tracker/specialist/weights/au_plate_detector.pt")
    if ours.exists():
        from ultralytics import YOLO
        models["ours"] = ("ultra",YOLO(str(ours)))

    models["hf_yolov11n"]=("ultra",ultralytics_model(
        "morsetechlab/yolov11-license-plate-detection","license-plate-finetune-v1n.pt"))
    models["hf_yolov8n"]=("ultra",ultralytics_model(
        "Koushim/yolov8-license-plate-detection","best.pt"))

    results={}
    for name,(kind,m) in models.items():
        results[name]=run_ultra(m,images,imgsz=768,conf=.05)

    results["hf_yolos_rego"]=run_yolos("nickmuchi/yolos-small-rego-plates-detection",images,threshold=.05)

    summary={k:stats(v) for k,v in results.items()}
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
    (a.out/"detections.json").write_text(json.dumps(results,indent=2))
    overlay_grid(images,results,a.out/"overlays.jpg")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
