#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

LETTERS=string.ascii_uppercase
DIGITS=string.digits

NSW_PATTERNS=[
    "LLDDLL","LLLDDL","LLLDDD","LLDDD","LLDDDD",
    "DDDLLL","DDLLLL","DDLLL",
]
# National heavy vehicle plates use black text on white with a blue sash.
# The public NHVR example FB23CA fits LLDDLL; generation stays deliberately broad.
NHV_PATTERNS=["LLDDLL","LLLDDD","DDDLLL"]

def rand_from(pattern:str)->str:
    out=[]
    for c in pattern:
        if c=="L": out.append(random.choice(LETTERS))
        elif c=="D": out.append(random.choice(DIGITS))
        else: out.append(c)
    return "".join(out)

def random_personalised()->str:
    n=random.randint(3,7)
    s="".join(random.choice(LETTERS+DIGITS) for _ in range(n))
    if s.isdigit() or s.isalpha():
        if random.random()<.65:
            k=random.randrange(n)
            s=s[:k]+random.choice(DIGITS if s.isalpha() else LETTERS)+s[k+1:]
    return s

def font(size:int):
    candidates=[
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        try:return ImageFont.truetype(p,size)
        except:pass
    return ImageFont.load_default()

def plate_base(kind:str,text:str)->Image.Image:
    if kind=="nsw_yellow":
        bg=(245,198,0); fg=(20,20,20); sash=None
    elif kind=="nhv":
        bg=(245,245,238); fg=(20,20,20); sash=(25,80,155)
    else:
        bg=(244,244,238); fg=(20,20,20); sash=None
    if kind=="nsw_auxiliary":
        w,h=(252,98)
    elif kind=="nsw_white":
        w,h=random.choice([(370,110),(375,90),(350,85)])
    else:
        w,h=random.choice([(370,130),(320,140),(370,110),(375,90),(350,85)])
    im=Image.new("RGB",(w,h),bg);d=ImageDraw.Draw(im)
    d.rounded_rectangle((3,3,w-4,h-4),radius=max(5,h//14),outline=(45,45,45),width=max(2,h//45))
    if sash:
        sh=max(15,int(h*.22));d.rectangle((3,h-sh,w-4,h-4),fill=sash)
        d.text((w//2,h-sh//2),"NATIONAL HEAVY VEHICLE",anchor="mm",font=font(max(7,sh//3)),fill=(255,255,255))
    elif kind=="nsw_auxiliary":
        d.text((w//2,5),"NSW – AUXILIARY",anchor="ma",font=font(max(8,h//10)),fill=fg)
    else:
        d.text((w//2,8),"NSW",anchor="ma",font=font(max(9,h//8)),fill=fg)
    top=int(h*.23);bottom=int(h*(.73 if sash else .86))
    available=max(20,bottom-top)
    fs=int(min(available*.9,w/(max(1,len(text))*0.72)))
    f=font(max(18,fs))
    d.text((w//2,(top+bottom)//2),text,anchor="mm",font=f,fill=fg,stroke_width=0)
    return im

def perspective(im:Image.Image)->Image.Image:
    w,h=im.size
    shear=random.uniform(-.13,.13)
    arr=np.array(im)
    # PIL affine keeps this dependency-light and sufficient for OCR domain randomisation.
    return im.transform((w,h),Image.Transform.AFFINE,(1,shear,-shear*h/2,random.uniform(-.03,.03),1,0),resample=Image.Resampling.BICUBIC,fillcolor=(30,30,30))

def degrade(im:Image.Image)->tuple[Image.Image,list[str]]:
    tags=[]
    if random.random()<.7:
        im=perspective(im);tags.append("oblique")
    if random.random()<.6:
        radius=random.uniform(.2,2.3);im=im.filter(ImageFilter.GaussianBlur(radius));tags.append("blur")
    if random.random()<.45:
        b=random.uniform(.25,.85);im=ImageEnhance.Brightness(im).enhance(b);tags.append("dark")
    if random.random()<.35:
        c=random.uniform(.55,1.5);im=ImageEnhance.Contrast(im).enhance(c);tags.append("contrast")
    if random.random()<.35:
        d=ImageDraw.Draw(im,"RGBA");w,h=im.size
        for _ in range(random.randint(1,4)):
            x=random.randint(0,w);y=random.randint(0,h);r=random.randint(max(4,h//18),max(8,h//4))
            d.ellipse((x-r,y-r,x+r,y+r),fill=(255,255,245,random.randint(25,100)))
        tags.append("glare")
    if random.random()<.35:
        d=ImageDraw.Draw(im,"RGBA");w,h=im.size
        for _ in range(random.randint(3,12)):
            x=random.randint(0,w);y=random.randint(0,h);rx=random.randint(2,max(3,w//20));ry=random.randint(1,max(2,h//15))
            d.ellipse((x-rx,y-ry,x+rx,y+ry),fill=(80,55,30,random.randint(15,75)))
        tags.append("dirt")
    if random.random()<.55:
        scale=random.uniform(.18,.75)
        small=im.resize((max(24,int(im.width*scale)),max(10,int(im.height*scale))),Image.Resampling.BILINEAR)
        im=small.resize(im.size,Image.Resampling.BICUBIC);tags.append("low_resolution")
    return im,tags

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("data/au_synth"))
    ap.add_argument("--count",type=int,default=10000)
    ap.add_argument("--seed",type=int,default=1404)
    args=ap.parse_args();random.seed(args.seed);np.random.seed(args.seed)
    imgdir=args.out/"images";imgdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    kinds=["nsw_white","nsw_yellow","nhv","nsw_auxiliary"]
    weights=[.42,.25,.25,.08]
    for i in range(args.count):
        kind=random.choices(kinds,weights)[0]
        if random.random()<.12:
            text=random_personalised();pattern="personalised"
        else:
            pats=NHV_PATTERNS if kind=="nhv" else NSW_PATTERNS
            pattern=random.choice(pats);text=rand_from(pattern)
        im=plate_base(kind,text);im,tags=degrade(im)
        fn=f"{i:08d}_{text}.jpg";im.save(imgdir/fn,quality=random.randint(55,94),subsampling=random.choice([0,1,2]))
        rows.append({"image":f"images/{fn}","text":text,"kind":kind,"pattern":pattern,"conditions":tags})
    split=int(len(rows)*.94)
    random.shuffle(rows)
    for name,part in [("train",rows[:split]),("val",rows[split:])]:
        (args.out/f"rec_gt_{name}.txt").write_text("".join(f"{r['image']}\t{r['text']}\n" for r in part),encoding="utf-8")
    (args.out/"manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows),encoding="utf-8")
    summary={"count":len(rows),"train":split,"val":len(rows)-split,"kinds":{},"conditions":{}}
    for r in rows:
        summary["kinds"][r["kind"]]=summary["kinds"].get(r["kind"],0)+1
        for t in r["conditions"]: summary["conditions"][t]=summary["conditions"].get(t,0)+1
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
