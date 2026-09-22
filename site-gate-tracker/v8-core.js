let Client=null,handle_file=null,gradioModulePromise=null;
async function ensureGradio(){
 if(Client&&handle_file)return true;
 if(!gradioModulePromise)gradioModulePromise=import("https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js").then(m=>{Client=m.Client;handle_file=m.handle_file;return true}).catch(e=>{gradioModulePromise=null;throw e});
 return gradioModulePromise;
}

const $=id=>document.getElementById(id);
const video=$('video'),canvas=$('overlay'),ctx=canvas.getContext('2d'),stage=$('stage');
const VEHICLE_IDS=new Set([2,3,5,7]);
const CLASS_NAME={2:'car',3:'motorcycle',5:'bus',7:'truck'};
const LABEL={car:'Light Vehicle',motorcycle:'Motorcycle',bus:'Bus',truck:'Heavy Vehicle'};
const YOLO26_URL='https://huggingface.co/besit/yolo-onnx/resolve/main/yolo26n.onnx?download=true';
const AU_VEHICLE_PROFILE_URL='./specialist/weights/au_vehicle_profile.json';
const AU_PLATE_MODEL_URL='./specialist/weights/au_plate_detector.onnx';
const AU_OCR_MODEL_URL='./specialist/weights/au_ocr_seed.onnx';
const AU_OCR_ALPHABET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const AU_CLASS_KEY={2:'light_vehicle',3:'motorcycle',5:'bus',7:'heavy_vehicle'};
let auVehicleProfile=null,auThreshold={2:.24,3:.24,5:.24,7:.24};
const ANPR_SPACE='Rickkosse/license-plate-detector';
const SESSION_ID=Date.now().toString(36).slice(-6).toUpperCase();
const freshDefaults=()=>({version:6,inside:0,inTotal:0,outTotal:0,events:[],line:null,waiting:null,reverse:false,dogGuard:true,paused:false,anpr:true});
let state=load();
let stream=null,running=false,detecting=false,detector=null,detectorName='—',inferCount=0,inferTotal=0,inferWindow=performance.now(),inferEMA=0,lastInfer=0,wakeLock=null;
let trackMap=new Map(),nextTrack=1,lastDets=[],recentHeavy=[],heavyPairVotes=new Map(),calMode=null,taps=[],lastUi=0;
let anprClient=null,anprConnectPromise=null,anprEndpoint=null,anprQueue=[],anprBusy=false,anprStatus='IDLE',anprFailures=0;
let auPlateModel=null,auOcrSession=null,localAnprLoading=null,localAnprStatus='IDLE',platePassMemory=new Map();

function load(){try{return Object.assign(freshDefaults(),JSON.parse(localStorage.getItem('siteGateTracker')||'{}'),{version:6})}catch{return freshDefaults()}}
function save(){localStorage.setItem('siteGateTracker',JSON.stringify(state));render()}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]))}
function heavy(name){return name==='truck'||name==='bus'}
function center(b){return[(b[0]+b[2])/2,(b[1]+b[3])/2]}
function area(b){return Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1])}
function iou(a,b){const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]);const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1),aa=area(a),bb=area(b);return inter/(aa+bb-inter||1)}
function linePx(){return state.line?state.line.map(p=>[p[0]*canvas.width,p[1]*canvas.height]):null}
function signedDist(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],len=Math.hypot(dx,dy)||1;return(dx*(p[1]-a[1])-dy*(p[0]-a[0]))/len}
function projT(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],den=dx*dx+dy*dy||1;return((p[0]-a[0])*dx+(p[1]-a[1])*dy)/den}
function insideWaiting(p){if(!state.waiting)return false;const[x1,y1,x2,y2]=state.waiting;return p[0]>=Math.min(x1,x2)*canvas.width&&p[0]<=Math.max(x1,x2)*canvas.width&&p[1]>=Math.min(y1,y2)*canvas.height&&p[1]<=Math.max(y1,y2)*canvas.height}
function shortId(id){const a=String(id).split('-');return a[a.length-1]}
function plateLabel(t){const p=t.plate||t.bestCandidate||'';const p2=t.plate2||t.bestCandidate2||'';return p+(p&&p2?' / ':'')+p2}

function render(){
 $('inside').textContent=state.inside;$('inTotal').textContent=state.inTotal;$('outTotal').textContent=state.outTotal;
 $('reverse').textContent='DIRECTION: '+(state.reverse?'REVERSED':'NORMAL');$('dogGuard').textContent='DOG GUARD: '+(state.dogGuard?'ON':'OFF');$('pause').textContent='COUNTING: '+(state.paused?'PAUSED':'LIVE');$('anpr').textContent='ANPR: '+(state.anpr?'ON':'OFF');
 $('detector').textContent=detectorName;$('anprHealth').textContent=state.anpr?(localAnprStatus+(anprStatus==='READY'?' +ADV':'')):'OFF';$('queue').textContent=anprQueue.length+(anprBusy?1:0);
 const active=[...trackMap.values()].filter(t=>performance.now()-t.lastSeen<3500).sort((a,b)=>b.lastSeen-a.lastSeen);
 $('tracks').textContent=active.length;$('waiting').textContent=active.filter(t=>insideWaiting([t.cx,t.cy])).length;
 const ib=$('identities');ib.innerHTML='';
 if(!active.length)ib.innerHTML='<div class="small">No active identities yet</div>';
 else active.slice(0,8).forEach(t=>{const d=document.createElement('div');d.className='idrow';const plate=plateLabel(t)||'observing';const typ=t.groupId?'Truck & Dog':(LABEL[t.label]||t.label);const st=t.groupId?((t.plate&&t.plate2)?'CONFIRMED':((t.plate||t.bestCandidate)?'PARTIAL':'GROUPING')):(t.plate?'CONFIRMED':(t.anprAttempts?'FUSING':'TRACKING'));d.innerHTML=`<b>#${esc(shortId(t.id))}</b><span>${esc(typ)}</span><span class="plate">${esc(plate)}</span><span>${esc(st)}</span>`;ib.appendChild(d)});
 $('identityDiag').textContent=active.filter(t=>t.plate).length+' confirmed';
 const eb=$('events');eb.innerHTML='';
 if(!state.events.length)eb.innerHTML='<div class="small" style="padding:12px;text-align:center">No movements yet</div>';
 state.events.slice(0,14).forEach(e=>{const d=document.createElement('div');d.className='event';const p=[e.plate,e.plate2].filter(Boolean).join('/');d.innerHTML=`<span>${esc(e.time)}</span><span>${esc(e.type)}</span><b>${esc(e.dir)}</b><span>#${e.inside}</span><span class="p">${esc(p||'—')}</span>`;eb.appendChild(d)});
 $('diag').textContent=detectorName+(inferEMA?' · '+Math.round(inferEMA)+'ms':'');
}
render();

async function keepAwake(){try{if('wakeLock'in navigator)wakeLock=await navigator.wakeLock.request('screen')}catch{}}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&running)keepAwake()});
function loadScript(src){return new Promise((res,rej)=>{const s=document.createElement('script');s.src=src;s.onload=res;s.onerror=rej;document.head.appendChild(s)})}

async function loadAUVehicleProfile(){
 try{
   const r=await fetch(AU_VEHICLE_PROFILE_URL,{cache:'no-store'});
   if(!r.ok)return;
   const p=await r.json();auVehicleProfile=p;
   for(const [id,key] of Object.entries(AU_CLASS_KEY)){
     const q=p?.geometry?.[key]?.confidence?.p10;
     if(Number.isFinite(q))auThreshold[id]=Math.max(.18,Math.min(.36,q*.88));
   }
 }catch(e){console.warn('AU vehicle profile unavailable; using baseline thresholds',e)}
}

async function ensureLocalANPR(){
 if(auPlateModel&&auOcrSession){localAnprStatus='READY';render();return true}
 if(localAnprLoading)return localAnprLoading;
 localAnprStatus='LOAD';render();
 localAnprLoading=(async()=>{
   try{
     const mod=await import('https://esm.sh/@ultralytics/yolo?bundle');
     auPlateModel=await mod.YOLO.load(AU_PLATE_MODEL_URL,{device:'auto'});
     try{auOcrSession=await ort.InferenceSession.create(AU_OCR_MODEL_URL,{executionProviders:['webgpu','wasm']})}
     catch(e){console.warn('AU OCR WebGPU unavailable, using WASM',e);auOcrSession=await ort.InferenceSession.create(AU_OCR_MODEL_URL,{executionProviders:['wasm']})}
     localAnprStatus='READY';render();return true;
   }catch(e){
     console.warn('Local AU ANPR unavailable',e);auPlateModel=null;auOcrSession=null;localAnprStatus='FALLBACK';render();return false;
   }finally{localAnprLoading=null}
 })();
 return localAnprLoading;
}
function plateInputCanvas(source){
 const W=160,H=48,c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d',{willReadFrequently:true});
 x.fillStyle='rgb(127,127,127)';x.fillRect(0,0,W,H);
 const sw=source.width||source.videoWidth||1,sh=source.height||source.videoHeight||1,s=Math.min(W/sw,H/sh),dw=Math.max(1,Math.round(sw*s)),dh=Math.max(1,Math.round(sh*s)),dx=Math.floor((W-dw)/2),dy=Math.floor((H-dh)/2);
 x.drawImage(source,0,0,sw,sh,dx,dy,dw,dh);
 const im=x.getImageData(0,0,W,H).data,a=new Float32Array(W*H);
 for(let i=0,j=0;i<a.length;i++,j+=4){const g=.299*im[j]+.587*im[j+1]+.114*im[j+2];a[i]=(g/255-.5)/.5}
 return a;
}
function decodeAUOCR(data,dims){
 let T=0,C=0,transposed=false;
 if(dims?.length===3){if(dims[2]===37){T=dims[1];C=dims[2]}else if(dims[1]===37){T=dims[2];C=dims[1];transposed=true}}
 if(!T||C!==37){C=37;T=Math.floor(data.length/C);transposed=false}
 let prev=-1,text='',logSum=0,n=0;
 for(let t=0;t<T;t++){
   let max=-Infinity,best=0;
   for(let k=0;k<C;k++){const v=transposed?Number(data[k*T+t]):Number(data[t*C+k]);if(v>max){max=v;best=k}}
   let sum=0;
   for(let k=0;k<C;k++){const v=transposed?Number(data[k*T+t]):Number(data[t*C+k]);sum+=Math.exp(v-max)}
   const prob=1/Math.max(1e-9,sum);
   if(best!==0&&best!==prev){text+=AU_OCR_ALPHABET[best-1]||'';logSum+=Math.log(Math.max(1e-6,prob));n++}
   prev=best;
 }
 return{text:normalizePlate(text),conf:n?Math.exp(logSum/n):0};
}
async function readAUPlateCanvas(cv){
 if(!auOcrSession)return{text:'',conf:0};
 const input=plateInputCanvas(cv),name=auOcrSession.inputNames[0],out=await auOcrSession.run({[name]:new ort.Tensor('float32',input,[1,1,48,160])}),tensor=out[auOcrSession.outputNames[0]]||out[Object.keys(out)[0]];
 return decodeAUOCR(tensor.data,tensor.dims||[]);
}
async function bitmapFromBlob(blob){
 if('createImageBitmap'in window)return await createImageBitmap(blob);
 const u=URL.createObjectURL(blob);try{const im=new Image();await new Promise((res,rej)=>{im.onload=res;im.onerror=rej;im.src=u});return im}finally{setTimeout(()=>URL.revokeObjectURL(u),1000)}
}
async function callLocalANPR(blob){
 if(!(await ensureLocalANPR()))return[];
 const image=await bitmapFromBlob(blob),r=await auPlateModel.predict(image,{conf:.18,iou:.55}),boxes=(r.boxes||[]).filter(b=>{
   const w=b.x2-b.x1,h=b.y2-b.y1,ar=w/Math.max(1,h),frac=(w*h)/Math.max(1,(r.width||image.width)*(r.height||image.height));
   return ar>1.15&&ar<8.5&&frac>.001&&frac<.45;
 }).sort((a,b)=>b.conf-a.conf).slice(0,2),reads=[];
 for(const b of boxes){
   const pad=Math.max(3,Math.round((b.y2-b.y1)*.18)),x=Math.max(0,Math.floor(b.x1-pad)),y=Math.max(0,Math.floor(b.y1-pad)),x2=Math.min(image.width,Math.ceil(b.x2+pad)),y2=Math.min(image.height,Math.ceil(b.y2+pad)),w=x2-x,h=y2-y;
   if(w<20||h<8)continue;
   const cv=document.createElement('canvas');cv.width=w;cv.height=h;cv.getContext('2d').drawImage(image,x,y,w,h,0,0,w,h);
   const o=await readAUPlateCanvas(cv);if(o.text.length>=3&&o.text.length<=10&&o.conf>=.25)reads.push({text:o.text,conf:Math.min(.995,o.conf*(.72+.28*Number(b.conf||0))),local:true,detConf:Number(b.conf||0)});
 }
 try{image.close?.()}catch{}
 return reads;
}

async function loadYOLO26(){
 if(!window.ort)throw new Error('ONNX Runtime unavailable');
 try{ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'}catch{}
 $('status').textContent='Loading YOLO26…';
 let session;
 try{session=await ort.InferenceSession.create(YOLO26_URL,{executionProviders:['webgpu','wasm']})}catch(e){console.warn('WebGPU load failed, trying WASM',e);session=await ort.InferenceSession.create(YOLO26_URL,{executionProviders:['wasm']})}
 const inputName=session.inputNames[0],outputName=session.outputNames[0],prep=document.createElement('canvas');
 prep.width=640;prep.height=640;const pctx=prep.getContext('2d',{willReadFrequently:true});
 detector={kind:'yolo26',predict:async()=>{
   const w=video.videoWidth,h=video.videoHeight,scale=Math.min(640/w,640/h),dw=Math.round(w*scale),dh=Math.round(h*scale),px=(640-dw)/2,py=(640-dh)/2;
   pctx.fillStyle='rgb(114,114,114)';pctx.fillRect(0,0,640,640);pctx.drawImage(video,0,0,w,h,px,py,dw,dh);
   const rgba=pctx.getImageData(0,0,640,640).data,n=640*640,x=new Float32Array(n*3);
   for(let i=0,j=0;i<n;i++,j+=4){x[i]=rgba[j]/255;x[n+i]=rgba[j+1]/255;x[2*n+i]=rgba[j+2]/255}
   const feeds={[inputName]:new ort.Tensor('float32',x,[1,3,640,640])},out=await session.run(feeds),o=out[outputName]||out[Object.keys(out)[0]],data=o.data,dims=o.dims||[];
   if(data.length%6!==0||!(dims[dims.length-1]===6||data.length<=6000))throw new Error('Unexpected YOLO26 output '+dims.join('x'));
   const rows=[],count=data.length/6;
   for(let i=0;i<count;i++){const k=i*6;let x1=Number(data[k]),y1=Number(data[k+1]),x2=Number(data[k+2]),y2=Number(data[k+3]),score=Number(data[k+4]),cls=Math.round(Number(data[k+5]));if(!VEHICLE_IDS.has(cls)||score<(auThreshold[cls]??.24))continue;if(Math.max(x1,y1,x2,y2)<=2){x1*=640;y1*=640;x2*=640;y2*=640}x1=(x1-px)/scale;y1=(y1-py)/scale;x2=(x2-px)/scale;y2=(y2-py)/scale;rows.push({name:CLASS_NAME[cls],score,bbox:[Math.max(0,x1),Math.max(0,y1),Math.min(w,x2),Math.min(h,y2)]})}
   return rows.filter(d=>area(d.bbox)>400);
 }};
 detectorName=auVehicleProfile?'YOLO26 · AU-calibrated':'YOLO26';$('status').textContent=(auVehicleProfile?'YOLO26 AU-calibrated':'YOLO26 local')+' · identity fusion';render();
}

async function loadSSD(){
 $('status').textContent='Loading fallback detector…';
 if(!window.tf)await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js');
 if(!window.cocoSsd)await loadScript('https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js');
 try{await tf.setBackend('webgl')}catch{}await tf.ready();const m=await cocoSsd.load({base:'lite_mobilenet_v2'});
 detector={kind:'ssd',predict:async()=>{const r=await m.detect(video,30,.32);return r.filter(x=>['car','truck','bus','motorcycle'].includes(x.class)).map(x=>({name:x.class,score:x.score,bbox:[x.bbox[0],x.bbox[1],x.bbox[0]+x.bbox[2],x.bbox[1]+x.bbox[3]]}))}};
 detectorName='SSD-FB';$('status').textContent='Fallback vision · identity fusion';render();
}
async function loadDetector(){await loadAUVehicleProfile();try{await loadYOLO26()}catch(e){console.warn('YOLO26 unavailable',e);await loadSSD()}}

async function start(){
 if(running)return;$('start').disabled=true;$('status').textContent='Requesting camera…';
 try{
   stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080},frameRate:{ideal:30,max:30}},audio:false});
   video.srcObject=stream;await video.play();await new Promise(r=>video.readyState>=2?r():video.addEventListener('loadedmetadata',r,{once:true}));resize();await loadDetector();running=true;keepAwake();$('start').textContent='CAMERA RUNNING';$('hint').textContent=state.line?'Tracking physical vehicles through the gate.':'Set the gate line with two taps.';requestAnimationFrame(loop);if(state.anpr)ensureLocalANPR().catch(()=>{});
 }catch(e){console.error(e);$('status').textContent='Camera/model error';$('hint').textContent='Could not start. Open in Safari over HTTPS and allow Camera.';$('start').disabled=false}
}
function resize(){if(video.videoWidth){canvas.width=video.videoWidth;canvas.height=video.videoHeight}}
video.addEventListener('loadedmetadata',resize);window.addEventListener('resize',resize);

function newTrack(d,now){const c=center(d.bbox),id=SESSION_ID+'-'+(nextTrack++);return{id,bbox:d.bbox,cx:c[0],cy:c[1],vx:0,vy:0,lastSeen:now,hits:1,score:d.score,label:d.name,votes:{[d.name]:d.score},stableSide:null,candidateSide:null,candidateCount:0,lastEvent:0,plateObs:[],plateObs2:[],plate:null,plate2:null,plateConfidence:0,plate2Confidence:0,bestCandidate:null,bestCandidate2:null,anprAttempts:0,lastAnpr:0,anprPending:false,bestAnprQ:0,groupId:null,groupMate:null}}
function updateTracks(dets,now){
 const active=[...trackMap.values()].filter(t=>now-t.lastSeen<3500),used=new Set();dets.sort((a,b)=>area(b.bbox)-area(a.bbox));
 for(const d of dets){const c=center(d.bbox),diag=Math.hypot(d.bbox[2]-d.bbox[0],d.bbox[3]-d.bbox[1]);let best=null,bestCost=99;
   for(const t of active){if(used.has(t.id)||heavy(t.label)!==heavy(d.name))continue;const dt=Math.min((now-t.lastSeen)/250,5),px=t.cx+t.vx*dt,py=t.cy+t.vy*dt,dist=Math.hypot(c[0]-px,c[1]-py),maxD=Math.max(100,diag*.9);if(dist>maxD*1.35)continue;const cost=dist/maxD+(1-iou(d.bbox,t.bbox))*.5;if(cost<bestCost){best=t;bestCost=cost}}
   if(!best||bestCost>1.55){best=newTrack(d,now);trackMap.set(best.id,best)}else{used.add(best.id);const ox=best.cx,oy=best.cy;best.vx=.72*best.vx+.28*(c[0]-ox);best.vy=.72*best.vy+.28*(c[1]-oy);best.cx=c[0];best.cy=c[1];best.bbox=d.bbox;best.score=d.score;best.lastSeen=now;best.hits++;best.votes[d.name]=(best.votes[d.name]||0)+Math.max(.2,d.score);best.label=Object.entries(best.votes).sort((a,b)=>b[1]-a[1])[0][0]}
   d.track=best;used.add(best.id);
 }
 for(const[id,t]of trackMap)if(now-t.lastSeen>15000&&!t.anprPending)trackMap.delete(id);
 return dets;
}

function pairKey(a,b){return [a.id,b.id].sort().join('|')}
function updateHeavyGroups(now){
 const a=[...trackMap.values()].filter(t=>heavy(t.label)&&now-t.lastSeen<1200&&t.hits>=3);
 const touched=new Set();
 for(let i=0;i<a.length;i++)for(let j=i+1;j<a.length;j++){
   const x=a[i],y=a[j],key=pairKey(x,y),dx=y.cx-x.cx,dy=y.cy-x.cy,dist=Math.hypot(dx,dy);
   const wx=Math.max(40,x.bbox[2]-x.bbox[0]),wy=Math.max(40,x.bbox[3]-x.bbox[1]),yx=Math.max(40,y.bbox[2]-y.bbox[0]),yy=Math.max(40,y.bbox[3]-y.bbox[1]);
   const scale=Math.max(Math.hypot(wx,wy),Math.hypot(yx,yy)),sx=Math.hypot(x.vx,x.vy),sy=Math.hypot(y.vx,y.vy);
   let aligned=true,cross=0;
   if(sx>2&&sy>2){const dot=(x.vx*y.vx+x.vy*y.vy)/(sx*sy);aligned=dot>.72;const ux=(x.vx/sx+y.vx/sy)/2,uy=(x.vy/sx+y.vy/sy)/2;cross=Math.abs(dx*uy-dy*ux)}
   const close=dist<scale*1.65&&dist>Math.min(wx,yx)*.28;
   const laneOk=(sx<=2||sy<=2)?Math.abs(dy)<scale*1.15:cross<scale*.72;
   if(aligned&&close&&laneOk){
     const v=(heavyPairVotes.get(key)||0)+1;heavyPairVotes.set(key,Math.min(12,v));touched.add(key);
     if(v>=3){
       const gid=x.groupId||y.groupId||('G-'+SESSION_ID+'-'+[shortId(x.id),shortId(y.id)].sort().join('-'));
       x.groupId=gid;y.groupId=gid;x.groupMate=y.id;y.groupMate=x.id;
       if(x.plate&&y.plate&&x.plate!==y.plate){x.plate2=x.plate2||y.plate;y.plate2=y.plate2||x.plate}
       if(x.bestCandidate&&y.bestCandidate&&x.bestCandidate!==y.bestCandidate){x.bestCandidate2=x.bestCandidate2||y.bestCandidate;y.bestCandidate2=y.bestCandidate2||x.bestCandidate}
     }
   }
 }
 for(const[k,v]of heavyPairVotes){if(!touched.has(k)){const n=v-1;if(n<=0)heavyPairVotes.delete(k);else heavyPairVotes.set(k,n)}}
}
function mergeGroupedEvent(t,dir){
 if(!t.groupId)return false;
 const mate=t.groupMate?trackMap.get(t.groupMate):null;
 const e=state.events.find(x=>x.vehicleGroupId===t.groupId&&x.dir===dir);
 if(!e)return false;
 e.type='Truck & Dog';e.trackIds=[...new Set([...(e.trackIds||[e.trackId]),t.id,mate?.id].filter(Boolean))];
 const plates=[e.plate,e.plate2,t.plate,t.plate2,t.bestCandidate,t.bestCandidate2,mate?.plate,mate?.plate2,mate?.bestCandidate].filter(Boolean);
 const unique=[...new Set(plates)];e.plate=unique[0]||'';e.plate2=unique[1]||'';e.grouped=true;save();return true
}

function checkCrossing(t,now){
 const ln=linePx();if(!ln||state.paused||t.hits<3)return;const[a,b]=ln,p=[t.cx,t.cy],dist=signedDist(p,a,b),margin=Math.max(12,canvas.height*.015);let side=0;if(dist>margin)side=1;else if(dist<-margin)side=-1;else return;
 if(t.stableSide===null){t.stableSide=side;return}if(side===t.stableSide){t.candidateSide=null;t.candidateCount=0;return}if(t.candidateSide===side)t.candidateCount++;else{t.candidateSide=side;t.candidateCount=1}if(t.candidateCount<2)return;
 const pr=projT(p,a,b);if(pr<-.12||pr>1.12){t.stableSide=side;t.candidateSide=null;t.candidateCount=0;return}if(now-t.lastEvent<1600){t.stableSide=side;t.candidateCount=0;return}
 let dir=(t.stableSide===-1&&side===1)?'IN':'OUT';if(state.reverse)dir=dir==='IN'?'OUT':'IN';
 const passPlates=[t.plate,t.plate2].filter(Boolean),wall=Date.now(),plateDup=passPlates.find(p=>wall-(platePassMemory.get(p+'|'+dir)||0)<10000);
 if(plateDup){$('diag').textContent='Plate duplicate suppressed · '+plateDup;t.stableSide=side;t.candidateSide=null;t.candidateCount=0;t.lastEvent=now;return}
 if(state.dogGuard&&heavy(t.label)){recentHeavy=recentHeavy.filter(x=>now-x.time<3000);const dup=recentHeavy.find(x=>x.dir===dir&&x.id!==t.id&&((t.groupId&&x.groupId===t.groupId&&now-x.time<2800)||(!t.groupId&&now-x.time<700)));if(dup){if(t.groupId&&dup.groupId===t.groupId)mergeGroupedEvent(t,dir);t.stableSide=side;t.candidateSide=null;t.candidateCount=0;t.lastEvent=now;return}recentHeavy.push({time:now,dir,id:t.id,groupId:t.groupId||null})}
 t.stableSide=side;t.candidateSide=null;t.candidateCount=0;t.lastEvent=now;addEvent(dir,LABEL[t.label]||t.label,'AI',t);for(const p of passPlates)platePassMemory.set(p+'|'+dir,wall);for(const[k,v]of platePassMemory)if(wall-v>30000)platePassMemory.delete(k);if(state.anpr&&!t.plate)scheduleANPR(t,now,true);
}
function addEvent(dir,type,source='AI',track=null){if(dir==='IN'){state.inside++;state.inTotal++}else{state.inside=Math.max(0,state.inside-1);state.outTotal++}const e={id:Date.now()+'-'+Math.random().toString(36).slice(2,7),time:new Date().toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}),iso:new Date().toISOString(),type,dir,inside:state.inside,source,trackId:track?.id||'',vehicleGroupId:track?.groupId||track?.id||'',trackIds:track?[track.id]:[],plate:track?.plate||track?.bestCandidate||'',plate2:track?.plate2||track?.bestCandidate2||'',plateConfidence:track?.plateConfidence||0,plate2Confidence:track?.plate2Confidence||0};state.events.unshift(e);state.events=state.events.slice(0,500);save();flash(dir,track?.plate||'')}
function backfillEvents(t){let changed=false;for(const e of state.events){if(e.trackId!==t.id)continue;if(t.plate&&e.plate!==t.plate){e.plate=t.plate;e.plateConfidence=t.plateConfidence;changed=true}if(t.plate2&&e.plate2!==t.plate2){e.plate2=t.plate2;e.plate2Confidence=t.plate2Confidence;changed=true}}if(changed)save()}
function flash(dir,plate=''){const f=$('flash');f.textContent=dir+(plate?' · '+plate:'');f.classList.add('show');setTimeout(()=>f.classList.remove('show'),900)}

function normalizePlate(s){return String(s||'').toUpperCase().replace(/[^A-Z0-9]/g,'').slice(0,10)}
const CONF=new Set(['0O','O0','1I','I1','1L','L1','2Z','Z2','5S','S5','6G','G6','8B','B8']);
function distance(a,b){if(a.length!==b.length)return 99;let d=0;for(let i=0;i<a.length;i++){if(a[i]===b[i])continue;d+=CONF.has(a[i]+b[i])?0.35:1}return d}
function fuseSlot(obs){if(!obs.length)return{best:null,confidence:0,confirmed:false};const candidates=[...new Set(obs.map(o=>o.text))],total=obs.reduce((s,o)=>s+o.weight,0)||1;let best=null,bestScore=-1,bestSupport=0;for(const c of candidates){let score=0,support=0;for(const o of obs){const d=distance(c,o.text),sim=d===0?1:(d<=.7?0.62:0);if(sim){score+=o.weight*sim;support++}}if(score>bestScore){best=c;bestScore=score;bestSupport=support}}const ratio=bestScore/total,confidence=Math.min(.99,ratio*(.72+Math.min(bestSupport,4)*.07)),confirmed=(bestSupport>=2&&ratio>=.56)||(bestSupport>=3&&ratio>=.48);return{best,confidence,confirmed}}
function recomputePlate(t){const a=fuseSlot(t.plateObs),b=fuseSlot(t.plateObs2);t.bestCandidate=a.best;t.plateConfidence=a.confidence;if(a.confirmed)t.plate=a.best;t.bestCandidate2=b.best;t.plate2Confidence=b.confidence;if(b.confirmed&&b.best!==t.plate)t.plate2=b.best;if(t.groupId&&t.groupMate){const m=trackMap.get(t.groupMate);if(m){if(t.plate&&m.plate&&t.plate!==m.plate){t.plate2=t.plate2||m.plate;m.plate2=m.plate2||t.plate}if(t.bestCandidate&&m.bestCandidate&&t.bestCandidate!==m.bestCandidate){t.bestCandidate2=t.bestCandidate2||m.bestCandidate;m.bestCandidate2=m.bestCandidate2||t.bestCandidate}}}backfillEvents(t)}
function parseANPR(text){const out=[];const s=String(text||'').toUpperCase();const re=/([A-Z0-9][A-Z0-9 -]{2,11})\s*\(OCR\s+(\d{1,3})%\)/g;let m;while((m=re.exec(s))){const p=normalizePlate(m[1]);if(p.length>=3&&p.length<=10&&!/PLATEDETECTED|UNREADABLE/.test(p))out.push({text:p,conf:Math.min(1,Number(m[2])/100)})}return out}

async function connectANPR(){
 if(anprClient)return anprClient;if(anprConnectPromise)return anprConnectPromise;anprStatus='WAKE';render();
 anprConnectPromise=(async()=>{try{await ensureGradio();const c=await Client.connect(ANPR_SPACE,{status_callback:s=>{if(s?.status==='sleeping'||s?.status==='building'){anprStatus='WAKE';render()}}});let ep='/run';try{const info=await c.view_api();const keys=Object.keys(info?.named_endpoints||{});ep=keys.find(k=>/run/i.test(k))||keys.find(k=>/predict/i.test(k))||keys[0]||'/run'}catch{}anprClient=c;anprEndpoint=ep;anprStatus='READY';anprFailures=0;render();return c}catch(e){anprStatus='ERR';anprConnectPromise=null;render();throw e}})();return anprConnectPromise;
}
async function cropTrack(t){
 if(!video.videoWidth)return null;const b=t.bbox,pad=.06*Math.max(b[2]-b[0],b[3]-b[1]),sx=Math.max(0,b[0]-pad),sy=Math.max(0,b[1]-pad),ex=Math.min(video.videoWidth,b[2]+pad),ey=Math.min(video.videoHeight,b[3]+pad),sw=ex-sx,sh=ey-sy;if(sw<100||sh<70)return null;const max=900,scale=Math.min(1,max/Math.max(sw,sh)),w=Math.max(1,Math.round(sw*scale)),h=Math.max(1,Math.round(sh*scale)),c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.drawImage(video,sx,sy,sw,sh,0,0,w,h);const q=Math.min(1,.35+.4*Math.min(1,area(b)/(canvas.width*canvas.height*.18))+.25*(t.score||0));const blob=await new Promise(r=>c.toBlob(r,'image/jpeg',.88));return blob?{blob,q,w,h}:null;
}
async function scheduleANPR(t,now=performance.now(),force=false){
 if(!state.anpr||t.plate||t.anprPending||t.anprAttempts>=4)return;if(!force&&(t.hits<3||now-t.lastAnpr<1400))return;const bw=t.bbox[2]-t.bbox[0],bh=t.bbox[3]-t.bbox[1];if(!force&&(bw<180||bh<100||area(t.bbox)<canvas.width*canvas.height*.018))return;t.anprPending=true;t.lastAnpr=now;
 try{const crop=await cropTrack(t);if(!crop){t.anprPending=false;return}if(!force&&t.bestAnprQ&&crop.q<t.bestAnprQ-.12){t.anprPending=false;return}t.bestAnprQ=Math.max(t.bestAnprQ||0,crop.q);t.anprAttempts++;if(anprQueue.length>=5){const old=anprQueue.shift();if(old?.t)old.t.anprPending=false}anprQueue.push({t,blob:crop.blob,q:crop.q,ts:Date.now()});render();pumpANPR()}catch{t.anprPending=false}
}
async function callANPR(blob){const c=await connectANPR();const eps=[anprEndpoint,'/run','/predict'].filter((x,i,a)=>x&&a.indexOf(x)===i);let last;for(const ep of eps){try{return await c.predict(ep,[handle_file(blob),.35])}catch(e){last=e}}throw last||new Error('ANPR endpoint unavailable')}
async function pumpANPR(){
 if(anprBusy||!state.anpr||!anprQueue.length)return;anprBusy=true;const job=anprQueue.shift(),t=job.t;anprStatus='LOCAL';render();
 try{
   let reads=[];try{reads=await callLocalANPR(job.blob)}catch(e){console.warn('Local AU ANPR pass failed',e)}
   const localStrong=reads.some(x=>x.conf>=.62);
   if(!localStrong){
     try{const result=await callANPR(job.blob),txt=(result?.data||[]).find(x=>typeof x==='string')||'',remote=parseANPR(txt);for(const r of remote)if(!reads.some(x=>x.text===r.text))reads.push({...r,local:false})}
     catch(e){console.warn('Advanced ANPR fallback unavailable',e)}
   }
   reads.sort((a,b)=>b.conf-a.conf);
   if(reads[0])t.plateObs.push({text:reads[0].text,conf:reads[0].conf,weight:Math.max(.05,reads[0].conf*job.q*(reads[0].local?1.12:1)),time:Date.now(),source:reads[0].local?'AU_LOCAL':'ADV'});
   if(reads[1]&&reads[1].text!==reads[0]?.text)t.plateObs2.push({text:reads[1].text,conf:reads[1].conf,weight:Math.max(.05,reads[1].conf*job.q),time:Date.now(),source:reads[1].local?'AU_LOCAL':'ADV'});
   t.plateObs=t.plateObs.slice(-16);t.plateObs2=t.plateObs2.slice(-16);recomputePlate(t);anprFailures=0;anprStatus=reads.length?'READY':'UNRES';
 }catch(e){console.warn('ANPR failed',e);anprFailures++;anprStatus=anprFailures>2?'ERR':'RETRY'}finally{t.anprPending=false;anprBusy=false;render();setTimeout(pumpANPR,220)}
}

async function infer(now){if(detecting||!detector)return;detecting=true;const t0=performance.now();try{const dets=await detector.predict();lastDets=updateTracks(dets,now);updateHeavyGroups(now);for(const d of lastDets){checkCrossing(d.track,now);scheduleANPR(d.track,now,false)}inferCount++;inferTotal++;const dt=performance.now()-t0;inferEMA=inferEMA?inferEMA*.82+dt*.18:dt;if(detector.kind==='yolo26'&&inferTotal>6&&inferEMA>2600){console.warn('YOLO26 too slow, switching to fallback');await loadSSD()}}catch(e){console.warn('Inference error',e);if(detector?.kind==='yolo26'){try{await loadSSD()}catch{}}}finally{detecting=false}}
function loop(now){if(!running)return;if(now-lastInfer>Math.max(180,Math.min(650,inferEMA*.45||250))){lastInfer=now;infer(now)}draw(now);if(now-inferWindow>1000){$('hz').textContent=inferCount;inferCount=0;inferWindow=now}if(now-lastUi>300){lastUi=now;render()}requestAnimationFrame(loop)}

function draw(now){
 ctx.clearRect(0,0,canvas.width,canvas.height);const ln=linePx();if(ln){ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(3,canvas.width/400);ctx.beginPath();ctx.moveTo(...ln[0]);ctx.lineTo(...ln[1]);ctx.stroke()}
 if(state.waiting){const[x1,y1,x2,y2]=state.waiting;ctx.save();ctx.strokeStyle='#f0c36a';ctx.lineWidth=3;ctx.setLineDash([14,9]);ctx.strokeRect(x1*canvas.width,y1*canvas.height,(x2-x1)*canvas.width,(y2-y1)*canvas.height);ctx.restore()}
 for(const t of trackMap.values()){if(now-t.lastSeen>1800)continue;const[x1,y1,x2,y2]=t.bbox;ctx.strokeStyle=t.plate?'#49d17d':'#4db2ff';ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);const p=plateLabel(t);const text=`#${shortId(t.id)} ${LABEL[t.label]||t.label}${p?' · '+(t.plate?'':'?')+p:''}`;ctx.font=`${Math.max(14,canvas.width/70)}px -apple-system,sans-serif`;const tw=ctx.measureText(text).width;ctx.fillStyle='#000c';ctx.fillRect(x1,Math.max(0,y1-25),tw+10,25);ctx.fillStyle='#fff';ctx.fillText(text,x1+5,Math.max(18,y1-7))}
}

function pointFromEvent(e){const r=canvas.getBoundingClientRect(),x=(e.clientX-r.left)/r.width,y=(e.clientY-r.top)/r.height;return[Math.max(0,Math.min(1,x)),Math.max(0,Math.min(1,y))]}
stage.addEventListener('pointerdown',e=>{if(!calMode)return;e.preventDefault();taps.push(pointFromEvent(e));if(taps.length<2){$('hint').textContent='Tap the second point.';return}if(calMode==='line'){state.line=[taps[0],taps[1]];$('hint').textContent='Gate line saved. Reverse direction if IN/OUT is backwards.'}else{state.waiting=[taps[0][0],taps[0][1],taps[1][0],taps[1][1]];$('hint').textContent='Waiting bay saved.'}calMode=null;taps=[];save()});

$('start').onclick=start;
$('gate').onclick=()=>{calMode='line';taps=[];$('hint').textContent='Tap two points across the vehicle path.'};
$('waitZone').onclick=()=>{calMode='waiting';taps=[];$('hint').textContent='Tap opposite corners of the waiting bay.'};
$('reverse').onclick=()=>{state.reverse=!state.reverse;save()};
$('dogGuard').onclick=()=>{state.dogGuard=!state.dogGuard;save()};
$('pause').onclick=()=>{state.paused=!state.paused;save()};
$('anpr').onclick=()=>{state.anpr=!state.anpr;if(!state.anpr){for(const j of anprQueue)j.t.anprPending=false;anprQueue=[];anprStatus='OFF';localAnprStatus='OFF'}else{anprStatus='IDLE';localAnprStatus='IDLE';if(running)ensureLocalANPR().catch(()=>{})}save()};
$('setInside').onclick=()=>{const n=prompt('Vehicles currently inside:',String(state.inside));if(n!==null&&Number.isFinite(Number(n))&&Number(n)>=0){state.inside=Math.floor(Number(n));save()}};
$('manualIn').onclick=()=>addEvent('IN',$('manualType').value,'MANUAL');
$('manualOut').onclick=()=>addEvent('OUT',$('manualType').value,'MANUAL');
$('undo').onclick=()=>{const e=state.events.shift();if(!e)return;if(e.dir==='IN'){state.inTotal=Math.max(0,state.inTotal-1);state.inside=Math.max(0,state.inside-1)}else{state.outTotal=Math.max(0,state.outTotal-1);state.inside++}save()};
$('capture').onclick=async()=>{const t=[...trackMap.values()].filter(x=>performance.now()-x.lastSeen<1800).sort((a,b)=>area(b.bbox)-area(a.bbox))[0];if(!t){$('hint').textContent='No active vehicle to capture.';return}const c=await cropTrack(t);if(!c)return;const a=document.createElement('a');a.href=URL.createObjectURL(c.blob);a.download=`vehicle-${t.id}-${t.plate||'unresolved'}.jpg`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};
$('export').onclick=()=>{const head=['time','iso','type','direction','inside','track_id','plate','secondary_plate','plate_confidence','secondary_confidence','source'];const rows=state.events.slice().reverse().map(e=>[e.time,e.iso,e.type,e.dir,e.inside,e.trackId||'',e.plate||'',e.plate2||'',e.plateConfidence||'',e.plate2Confidence||'',e.source||'']);const csv=[head,...rows].map(r=>r.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n');const b=new Blob([csv],{type:'text/csv'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='site-gate-'+new Date().toISOString().slice(0,10)+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)};
$('reset').onclick=()=>{if(!confirm('Reset this shift? Counts, movements and plate memory will be cleared.'))return;for(const j of anprQueue)j.t.anprPending=false;state=freshDefaults();trackMap.clear();anprQueue=[];recentHeavy=[];platePassMemory.clear();heavyPairVotes.clear();localStorage.removeItem('siteGateTracker');save()};

function operationalPhase(t){
 if(insideWaiting([t.cx,t.cy]))return 'WAITING';
 const ln=linePx();if(!ln)return 'TRACKING';const[a,b]=ln,d=signedDist([t.cx,t.cy],a,b),m=Math.max(12,canvas.height*.015);if(Math.abs(d)<=m)return 'GATE';
 const side=d>0?1:-1,outside=state.reverse?1:-1;return side===outside?'APPROACH':'SITE';
}
window.GateTrackerAPI={
 version:8,
 active:()=>[...trackMap.values()].filter(t=>performance.now()-t.lastSeen<2500).map(t=>({id:t.id,shortId:shortId(t.id),bbox:[...t.bbox],label:t.label,type:t.groupId?'Truck & Dog':(LABEL[t.label]||t.label),groupId:t.groupId||'',groupMate:t.groupMate||'',plate:t.plate||'',plate2:t.plate2||'',candidate:t.bestCandidate||'',candidate2:t.bestCandidate2||'',plateConfidence:t.plateConfidence||0,plate2Confidence:t.plate2Confidence||0,hits:t.hits||0,score:t.score||0,phase:operationalPhase(t),anprAttempts:t.anprAttempts||0})),
 crop:async id=>{const t=[...trackMap.values()].find(x=>x.id===id||shortId(x.id)===String(id));if(!t)return null;const c=await cropTrack(t);return c?.blob||null},
 state:()=>JSON.parse(JSON.stringify(state)),
 localAnpr:()=>({status:localAnprStatus,plateModel:!!auPlateModel,ocr:!!auOcrSession})
};

render();
