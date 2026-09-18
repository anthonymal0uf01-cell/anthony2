#!/usr/bin/env python3
from __future__ import annotations
import argparse, io, json, urllib.request
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

TFNSW="https://data.livetraffic.com/cameras/traffic-cam.json"
COCO={2:"light_vehicle",3:"motorcycle",5:"bus",7:"heavy_vehicle"}

def req(url,timeout=20):
    headers={
        "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1",
        "Accept":"image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8" if "webcams.transport.nsw.gov.au" in url else "application/json,text/plain,*/*",
        "Referer":"https://www.livetraffic.com/",
        "Accept-Language":"en-AU,en;q=0.9",
        "Cache-Control":"no-cache",
    }
    r=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(r,timeout=timeout) as x:return x.read()

def urls(node,out=None):
    if out is None: out=[]
    if isinstance(node,dict):
        for k,v in node.items():
            if isinstance(v,str) and v.startswith("http") and (any(z in k.lower() for z in ("image","href","url")) or any(z in v.lower() for z in (".jpg",".jpeg","camera","webcam","livetraffic"))):
                out.append(v)
            else: urls(v,out)
    elif isinstance(node,list):
        for x in node:urls(x,out)
    return list(dict.fromkeys(out))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--limit",type=int,default=400);args=ap.parse_args()
    from ultralytics import YOLO
    js=json.loads(req(TFNSW)); all_urls=urls(js); us=all_urls[:args.limit]
    model=YOLO("yolo26n.pt")
    counts=Counter();geom=defaultdict(list);frames=0
    for u in us:
        try:
            im=Image.open(io.BytesIO(req(u,10))).convert("RGB")
            r=model.predict(source=np.asarray(im),imgsz=640,conf=.25,verbose=False)[0]
            frames+=1
            if r.boxes is None:continue
            xyxy=r.boxes.xyxy.cpu().numpy();cl=r.boxes.cls.cpu().numpy().astype(int);cf=r.boxes.conf.cpu().numpy()
            W,H=im.size
            for b,k,q in zip(xyxy,cl,cf):
                if k not in COCO:continue
                name=COCO[k];counts[name]+=1
                x1,y1,x2,y2=b;geom[name].append({"area":float((x2-x1)*(y2-y1)/(W*H)),"aspect":float((x2-x1)/max(1,y2-y1)),"confidence":float(q)})
        except Exception:pass
    total=sum(counts.values()) or 1
    profile={"source":"TfNSW Live Traffic Cameras","license":"Creative Commons Attribution",
             "available_camera_urls":len(all_urls),"requested_limit":args.limit,"frames":frames,
             "detections":sum(counts.values()),"visual_prior":{k:v/total for k,v in counts.items()},"geometry":{}}
    for k,rows in geom.items():
        for feat in ("area","aspect","confidence"):
            a=np.array([r[feat] for r in rows],dtype=float)
            profile["geometry"].setdefault(k,{})[feat]={"p10":float(np.quantile(a,.1)),"median":float(np.median(a)),"p90":float(np.quantile(a,.9))}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(profile,indent=2),encoding="utf-8")
    print(json.dumps(profile,indent=2))
    if frames<50:raise SystemExit("insufficient live Australian frames for vehicle calibration")
    if sum(counts.values())<100:raise SystemExit("insufficient Australian vehicle detections for calibration")
if __name__=="__main__":main()
