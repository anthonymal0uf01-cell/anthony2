#!/usr/bin/env python3
"""
Build an Australian tiny-plate detector dataset on REAL TfNSW vehicle crops.

Why:
  Full-frame training shrinks a 20-35 px plate again when YOLO resizes the 800x600
  traffic frame. Live inference instead detects vehicles first and then runs the
  plate detector on an enlarged vehicle crop. This builder matches that inference
  geometry.

Truth:
  Background/vehicle appearance is real TfNSW imagery.
  Plate pixels/text are synthetic and therefore exact.
  Existing real registration regions are blurred before synthetic placement so they
  are not accidentally trained as unlabeled negatives.
"""
from __future__ import annotations
import argparse, io, json, random, string, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

TFNSW="https://data.livetraffic.com/cameras/traffic-cam.json"
LETTERS=string.ascii_uppercase
DIGITS=string.digits
PATTERNS=["LLDDLL","LLLDDL","LLLDDD","DDDLLL","LLDDD","LLDDDD"]

def req(url,timeout=25):
    headers={
      "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 Version/26.0 Mobile/15E148 Safari/604.1",
      "Accept":"image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8" if "webcams.transport.nsw.gov.au" in url else "application/json,text/plain,*/*",
      "Referer":"https://www.livetraffic.com/",
      "Accept-Language":"en-AU,en;q=0.9",
      "Cache-Control":"no-cache",
    }
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=timeout) as r:
        return r.read()

def camera_urls():
    j=json.loads(req(TFNSW))
    out=[]
    for f in j.get("features",[]):
        p=f.get("properties") or {}
        u=p.get("href")
        if u:
            out.append((u,{"id":f.get("id"),"title":p.get("title"),"view":p.get("view"),"region":p.get("region")}))
    return out

def text_for(pattern):
    return "".join(random.choice(LETTERS if c=="L" else DIGITS) for c in pattern)

def plate_image(text, style="nsw"):
    # Keep the rendering intentionally simple; geometry/domain corruption matters more
    # than decorative fidelity for this tiny-plate detector stage.
    w,h=220,62
    if style=="nhv":
        bg=(245,245,238); fg=(28,28,28); border=(40,40,40)
    else:
        bg=(246,246,240); fg=(25,25,25); border=(35,35,35)
    im=Image.new("RGB",(w,h),bg); d=ImageDraw.Draw(im)
    d.rectangle((1,1,w-2,h-2),outline=border,width=3)
    try:
        font=ImageFont.truetype("DejaVuSans-Bold.ttf",39)
    except Exception:
        font=ImageFont.load_default()
    box=d.textbbox((0,0),text,font=font)
    tw=box[2]-box[0]; th=box[3]-box[1]
    d.text(((w-tw)//2,(h-th)//2-3),text,fill=fg,font=font)
    return im

def perspectiveish(p):
    # Mild affine/perspective-like distortion through resize/shear surrogate that works
    # with Pillow only; tiny camera plates rarely preserve perfect rectangle geometry.
    w,h=p.size
    if random.random()<.55:
        sx=random.uniform(.82,1.18)
        p=p.resize((max(8,int(w*sx)),h),Image.Resampling.BICUBIC)
    if random.random()<.65:
        p=p.rotate(random.uniform(-5.5,5.5),resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(128,128,128))
    return p

def blur_real_plate_zone(crop):
    W,H=crop.size
    # Conservative probable front/rear registration zone.
    x1=int(W*.18);x2=int(W*.82);y1=int(H*.52);y2=int(H*.94)
    if x2<=x1 or y2<=y1:return crop
    patch=crop.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(max(2.0,W/35)))
    out=crop.copy();out.paste(patch,(x1,y1,x2,y2));return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--count",type=int,default=8000)
    ap.add_argument("--camera-scan",type=int,default=120)
    ap.add_argument("--seed",type=int,default=260922)
    ap.add_argument("--vehicle-model",default="yolo26n.pt")
    ap.add_argument("--min-vehicle-width",type=int,default=70)
    ap.add_argument("--min-vehicle-height",type=int,default=42)
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed)
    args.out.mkdir(parents=True,exist_ok=True)

    from ultralytics import YOLO
    vm=YOLO(args.vehicle_model)

    pool=[]
    cams=camera_urls()
    random.shuffle(cams)
    for url,meta in cams[:args.camera_scan]:
        try:
            frame=Image.open(io.BytesIO(req(url,12))).convert("RGB")
            r=vm.predict(source=np.asarray(frame),imgsz=640,conf=.20,iou=.55,verbose=False)[0]
            if r.boxes is None:continue
            names=r.names
            for box,cf,cl in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
                name=str(names[int(cl)]).lower()
                if name not in {"car","truck","bus","motorcycle"}:continue
                x1,y1,x2,y2=map(float,box)
                w=x2-x1;h=y2-y1
                if w<args.min_vehicle_width or h<args.min_vehicle_height:continue
                pad=max(3,.04*max(w,h))
                crop=frame.crop((max(0,int(x1-pad)),max(0,int(y1-pad)),min(frame.width,int(x2+pad)),min(frame.height,int(y2+pad))))
                if crop.width>=args.min_vehicle_width and crop.height>=args.min_vehicle_height:
                    pool.append((crop,meta,url,name,float(cf)))
        except Exception:
            continue

    if len(pool)<20:
        raise SystemExit(f"Too few live vehicle crops: {len(pool)}")

    manifest=[]
    counts={"train":0,"val":0,"test":0}
    for i in range(args.count):
        split="train" if i<int(args.count*.82) else ("val" if i<int(args.count*.92) else "test")
        idir=args.out/split/"images"; ldir=args.out/split/"labels"
        idir.mkdir(parents=True,exist_ok=True);ldir.mkdir(parents=True,exist_ok=True)

        base,meta,url,vclass,vconf=random.choice(pool)
        im=blur_real_plate_zone(base)
        W,H=im.size

        # Usually one registration plate per visible vehicle. A small fraction are clean
        # negatives after blurring the probable real plate region.
        is_negative=random.random()<.08
        labels=[]; objects=[]
        if not is_negative:
            txt=text_for(random.choice(PATTERNS))
            p=plate_image(txt,"nhv" if vclass in {"truck","bus"} and random.random()<.45 else "nsw")
            # Match observed hard regime: tiny plate is commonly ~12-30% of vehicle width.
            native_w=int(np.clip(W*random.uniform(.12,.30),14,72))
            native_h=max(5,int(native_w*(p.height/p.width)*random.uniform(.80,1.12)))
            p=p.resize((native_w,native_h),Image.Resampling.LANCZOS)
            p=perspectiveish(p)

            # Place in lower vehicle body. Trucks/buses get wider vertical freedom.
            pw,ph=p.size
            cx=int(W*random.uniform(.30,.70))
            cy=int(H*random.uniform(.62,.84 if vclass not in {"truck","bus"} else .90))
            x=max(0,min(W-pw,cx-pw//2))
            y=max(0,min(H-ph,cy-ph//2))

            # Reproduce sensor/compression degradation at native plate pixels.
            if random.random()<.72:
                p=p.filter(ImageFilter.GaussianBlur(random.uniform(.20,.85)))
            if random.random()<.45:
                p=ImageEnhance.Brightness(p).enhance(random.uniform(.62,1.28))

            im.paste(p,(x,y))
            xc=(x+pw/2)/W; yc=(y+ph/2)/H
            labels=[f"0 {xc:.6f} {yc:.6f} {pw/W:.6f} {ph/H:.6f}"]
            objects=[{"text":txt,"bbox":[x,y,pw,ph],"native_plate_width":pw,"native_plate_height":ph}]

        # Camera-like JPEG degradation after compositing.
        if random.random()<.75:
            b=io.BytesIO()
            im.save(b,format="JPEG",quality=random.randint(35,78),subsampling=2)
            im=Image.open(io.BytesIO(b.getvalue())).convert("RGB")

        fn=f"vehicle_plate_{i:06d}.jpg"
        im.save(idir/fn,quality=random.randint(78,94))
        (ldir/(Path(fn).stem+".txt")).write_text(("\n".join(labels)+"\n") if labels else "",encoding="utf-8")
        counts[split]+=1
        manifest.append({
          "file":f"{split}/images/{fn}","source_url":url,"camera":meta,
          "vehicle_class":vclass,"vehicle_confidence":vconf,
          "source_license":"TfNSW CC BY","objects":objects,
          "negative":is_negative,"curriculum":"vehicle_crop_tiny_plate"
        })

    (args.out/"data.yaml").write_text(
      f"path: {args.out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: license_plate\n",
      encoding="utf-8")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(x) for x in manifest),encoding="utf-8")
    (args.out/"summary.json").write_text(json.dumps({
      "vehicle_crop_pool":len(pool),"generated":counts,
      "source":"TfNSW Live Traffic Cameras","license":"CC BY",
      "purpose":"match live inference geometry: vehicle crop -> enlarged tiny plate detector"
    },indent=2),encoding="utf-8")
    print(json.dumps({"vehicle_crop_pool":len(pool),"generated":counts}))

if __name__=="__main__":
    main()
