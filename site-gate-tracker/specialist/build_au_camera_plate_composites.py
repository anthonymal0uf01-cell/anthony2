#!/usr/bin/env python3
from __future__ import annotations

import argparse, io, json, random, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZ";DIGITS="0123456789"
PATTERNS=["LLDDLL","LLLDDL","LLLDDD","DDDLLL","LLDDD","LLDDDD"]
TFNSW="https://data.livetraffic.com/cameras/traffic-cam.json"

def req(url,timeout=30):
    headers={
        "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1",
        "Accept":"image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8" if "webcams.transport.nsw.gov.au" in url else "application/json,text/plain,*/*",
        "Referer":"https://www.livetraffic.com/",
        "Accept-Language":"en-AU,en;q=0.9",
        "Cache-Control":"no-cache",
    }
    r=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(r,timeout=timeout) as x:return x.read()

def font(sz):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf","/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"]:
        try:return ImageFont.truetype(p,sz)
        except:pass
    return ImageFont.load_default()

def text_for(pat):
    return "".join(random.choice(LETTERS if c=="L" else DIGITS) for c in pat)

def plate():
    kind=random.choices(["nsw_white","nsw_yellow","nhv"],[.52,.28,.20])[0]
    text=text_for(random.choice(PATTERNS))
    w,h=random.choice([(180,58),(200,64),(220,58),(170,48)])
    bg=(243,243,238) if kind!="nsw_yellow" else (244,196,0)
    im=Image.new("RGB",(w,h),bg);d=ImageDraw.Draw(im)
    d.rounded_rectangle((1,1,w-2,h-2),radius=4,outline=(30,30,30),width=2)
    if kind=="nhv":
        d.rectangle((2,h-14,w-3,h-3),fill=(26,76,150))
    else:
        d.text((w//2,3),"NSW",anchor="ma",font=font(8),fill=(15,15,15))
    d.text((w//2,h//2-1),text,anchor="mm",font=font(max(20,int(h*.55))),fill=(15,15,15))
    if random.random()<.55: im=im.filter(ImageFilter.GaussianBlur(random.uniform(.1,1.1)))
    if random.random()<.35: im=ImageEnhance.Brightness(im).enhance(random.uniform(.45,.85))
    return im,text,kind

def parse_cameras(raw):
    j=json.loads(raw)
    out=[];seen=set()
    def walk(node, context=None):
        context=context or {}
        if isinstance(node,dict):
            local=dict(context)
            for k,v in node.items():
                if isinstance(v,(str,int,float,bool)) and k.lower() in {"title","name","view","description","region","road","suburb"}:
                    local[k]=v
            for k,v in node.items():
                if isinstance(v,str) and v.startswith("http"):
                    kl=k.lower();vl=v.lower()
                    if any(x in kl for x in ("image","camera","url","link","href")) or any(x in vl for x in (".jpg",".jpeg","camera","webcam","livetraffic")):
                        if v not in seen:
                            seen.add(v);out.append((v,local))
                else:
                    walk(v,local)
        elif isinstance(node,list):
            for x in node: walk(x,context)
    walk(j,{})
    # Prefer URLs that look image-like; retain broad fallback for changing TfNSW schema.
    imageish=[x for x in out if any(z in x[0].lower() for z in (".jpg",".jpeg","image","camera"))]
    return imageish or out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--count",type=int,default=800)
    ap.add_argument("--seed",type=int,default=20260918)
    args=ap.parse_args();random.seed(args.seed);np.random.seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)
    cams=parse_cameras(req(TFNSW))
    if not cams: raise SystemExit("No TfNSW camera image URLs parsed")
    cache=[]
    for url,meta in random.sample(cams,min(len(cams),120)):
        try:
            im=Image.open(io.BytesIO(req(url,12))).convert("RGB")
            if im.width>=320 and im.height>=180:cache.append((im,meta,url))
        except Exception:pass
    if len(cache)<8: raise SystemExit(f"Only {len(cache)} live camera images downloaded")
    # Detect real vehicles once per Australian camera frame. Synthetic plates are then
    # attached to physical vehicle boxes instead of arbitrary road coordinates.
    vehicle_model=None
    try:
        from ultralytics import YOLO
        vehicle_model=YOLO("yolo26n.pt")
    except Exception as e:
        print("VEHICLE_ATTACHMENT_FALLBACK",e)
    enriched=[]
    for im,meta,url in cache:
        boxes=[]
        if vehicle_model is not None:
            try:
                rr=vehicle_model.predict(source=np.asarray(im),imgsz=640,conf=.23,verbose=False)[0]
                if rr.boxes is not None:
                    xy=rr.boxes.xyxy.cpu().numpy();cl=rr.boxes.cls.cpu().numpy().astype(int)
                    for b,k in zip(xy,cl):
                        if k not in {2,5,7}: continue
                        x1,y1,x2,y2=map(float,b)
                        if (x2-x1)*(y2-y1) < im.width*im.height*.002: continue
                        boxes.append((x1,y1,x2,y2,k))
            except Exception: pass
        enriched.append((im,meta,url,boxes))
    cache=enriched
    manifest=[];ocr_rows={"train":[],"val":[],"test":[]};ocr_manifest=[]
    for i in range(args.count):
        split="train" if i<int(args.count*.82) else ("val" if i<int(args.count*.92) else "test")
        imgs=args.out/split/"images";labs=args.out/split/"labels";imgs.mkdir(parents=True,exist_ok=True);labs.mkdir(parents=True,exist_ok=True)
        base,meta,url,vehicle_boxes=random.choice(cache); im=base.copy();W,H=im.size
        # Place plates where vehicles are most likely in fixed traffic cameras:
        # lower/middle road field, with scale tied to scene depth.
        n=1 if random.random()<.78 else 2
        labels=[];objects=[]
        for k in range(n):
            p,txt,kind=plate()
            target=random.choice(vehicle_boxes) if vehicle_boxes else None
            if target:
                vx1,vy1,vx2,vy2,vcls=target
                vw=max(20.0,vx2-vx1);vh=max(16.0,vy2-vy1)
                # Registration plates normally occupy a small central lower-body region.
                tw=int(np.clip(vw*random.uniform(.20,.42),30,180))
                th=max(10,int(tw*(p.height/p.width)*random.uniform(.85,1.12)))
                cx=random.uniform(vx1+vw*.35,vx1+vw*.65)
                cy=random.uniform(vy1+vh*.61,vy1+vh*.84)
            else:
                cy=random.uniform(.55,.88)*H
                tw=int(np.clip(W*random.uniform(.045,.095),30,150))
                th=max(10,int(tw*(p.height/p.width)))
                cx=random.uniform(.10,.90)*W
            p=p.resize((tw,th),Image.Resampling.BICUBIC)
            angle=random.uniform(-8,8);p=p.rotate(angle,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(80,80,80))
            x=int(cx-p.width/2);y=int(cy-p.height/2)
            x=max(0,min(W-p.width,x));y=max(0,min(H-p.height,y))
            # blur a slightly larger patch first to reduce accidental old-plate leakage
            pad=max(2,int(p.width*.08));box=(max(0,x-pad),max(0,y-pad),min(W,x+p.width+pad),min(H,y+p.height+pad))
            patch=im.crop(box).filter(ImageFilter.GaussianBlur(max(1.2,p.width/60)))
            im.paste(patch,box);im.paste(p,(x,y))
            # Save a camera-domain OCR crop with surrounding pixels and final JPEG compression.
            crop_pad=max(2,int(p.width*.12))
            crop_box=(max(0,x-crop_pad),max(0,y-crop_pad),min(W,x+p.width+crop_pad),min(H,y+p.height+crop_pad))
            crop=im.crop(crop_box)
            ocr_dir=args.out/"ocr"/"images";ocr_dir.mkdir(parents=True,exist_ok=True)
            ocfn=f"{i:06d}_{k}_{txt}.jpg";crop.save(ocr_dir/ocfn,quality=random.randint(68,92))
            ocr_rows[split].append((f"images/{ocfn}",txt))
            ocr_manifest.append({"image":f"images/{ocfn}","text":txt,"kind":kind,"conditions":["tfnsw_camera_domain"],"split":split})
            xc=(x+p.width/2)/W;yc=(y+p.height/2)/H
            labels.append(f"0 {xc:.6f} {yc:.6f} {p.width/W:.6f} {p.height/H:.6f}")
            objects.append({"text":txt,"kind":kind,"bbox":[x,y,p.width,p.height]})
        # Keep some domain negatives, but blur probable registration regions first so
        # an unlabelled real plate is not trained as background.
        if random.random()<.10:
            for vx1,vy1,vx2,vy2,_ in vehicle_boxes:
                vw=max(1,int(vx2-vx1));vh=max(1,int(vy2-vy1))
                bx1=max(0,int(vx1+vw*.20));bx2=min(W,int(vx2-vw*.20))
                by1=max(0,int(vy1+vh*.55));by2=min(H,int(vy2))
                if bx2>bx1 and by2>by1:
                    patch=im.crop((bx1,by1,bx2,by2)).filter(ImageFilter.GaussianBlur(max(3,vw/18)))
                    im.paste(patch,(bx1,by1,bx2,by2))
            labels=[];objects=[]
        fn=f"au_scene_{i:06d}.jpg";im.save(imgs/fn,quality=random.randint(76,94))
        (labs/(Path(fn).stem+".txt")).write_text(("\n".join(labels)+"\n") if labels else "")
        manifest.append({"file":f"{split}/images/{fn}","camera":meta.get("title") or meta.get("name") or meta.get("view"),"source_url":url,"source_license":"TfNSW CC BY","objects":objects})
    (args.out/"data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n",encoding="utf-8")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest),encoding="utf-8")
    ocr_root=args.out/"ocr"
    for sp in ("train","val","test"):
        (ocr_root/f"rec_gt_{sp}.txt").write_text("".join(f"{p}\t{t}\n" for p,t in ocr_rows[sp]),encoding="utf-8")
    (ocr_root/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in ocr_manifest),encoding="utf-8")
    print(json.dumps({"images":len(manifest),"ocr_crops":sum(len(v) for v in ocr_rows.values()),"live_camera_backgrounds":len(cache),"source":"TfNSW Live Traffic Cameras","license":"CC BY"},indent=2))
if __name__=="__main__":main()
