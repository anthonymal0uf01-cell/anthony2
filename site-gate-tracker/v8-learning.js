const $=id=>document.getElementById(id);
const video=$('video');
const DB_NAME='site-gate-specialist-v1', STORE='samples';
let db=null,lastCapture=0,activeTracks=new Map(),specialistUrl=localStorage.getItem('gateSpecialistUrl')||'',advancedBusy=false,burstBusy=new Set();

function openDB(){return new Promise((resolve,reject)=>{const r=indexedDB.open(DB_NAME,1);r.onupgradeneeded=()=>{const d=r.result;if(!d.objectStoreNames.contains(STORE)){const s=d.createObjectStore(STORE,{keyPath:'id'});s.createIndex('trackId','trackId');s.createIndex('label','label');s.createIndex('createdAt','createdAt')}};r.onsuccess=()=>{db=r.result;resolve(db)};r.onerror=()=>reject(r.error)})}
function tx(mode='readonly'){return db.transaction(STORE,mode).objectStore(STORE)}
function put(x){return new Promise((res,rej)=>{const r=tx('readwrite').put(x);r.onsuccess=()=>res();r.onerror=()=>rej(r.error)})}
function all(){return new Promise((res,rej)=>{const r=tx().getAll();r.onsuccess=()=>res(r.result||[]);r.onerror=()=>rej(r.error)})}
function del(id){return new Promise((res,rej)=>{const r=tx('readwrite').delete(id);r.onsuccess=()=>res();r.onerror=()=>rej(r.error)})}

function injectUI(){
 const side=document.querySelector('.side')||document.body;const box=document.createElement('div');box.className='identity';box.id='specialistPanel';box.innerHTML=`
 <div class="identityHead"><span>GATE SPECIALIST MEMORY</span><span id="learnState">starting</span></div>
 <div class="idrow"><b>CASES</b><span id="caseCount">0</span><span id="labelCount">0 labelled</span><span id="hardCount">0 review</span></div>
 <div class="controls" style="margin-top:7px">
   <button id="labelLatest">LABEL LATEST HARD CASE</button><button id="setSpecialist">ADVANCED RECOGNISER</button>
   <button id="exportLearning">EXPORT TRAINING ZIP</button><button id="clearLearning" class="warn">CLEAR LEARNING MEMORY</button>
 </div>
 <div class="small" id="learnHint">Active learning keeps only diverse, high-value cases. Hard/partial identities get short multi-frame evidence bursts; easy duplicates are discarded.</div>`;side.appendChild(box);
 $('labelLatest').onclick=labelLatest;$('setSpecialist').onclick=setSpecialist;$('exportLearning').onclick=exportZip;$('clearLearning').onclick=clearAll;
}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot',"'":'&#039;'}[c]))}
function normalizePlate(s){return String(s||'').toUpperCase().replace(/[^A-Z0-9]/g,'').slice(0,10)}

function coreTracks(){
 const api=window.GateTrackerAPI;
 if(api?.active){
   const out=new Map();
   for(const t of api.active()){
     const grouped=!!t.groupId,status=grouped?((t.plate&&t.plate2)?'CONFIRMED':((t.plate||t.candidate)?'PARTIAL':'GROUPING')):(t.plate?'CONFIRMED':'TRACKING');
     out.set(t.id,{id:t.id,shortId:t.shortId||t.id,type:t.type||'',plate:t.plate||'',plate2:t.plate2||'',candidate:t.candidate||'',candidate2:t.candidate2||'',status,phase:t.phase||'TRACKING',groupId:t.groupId||'',groupMate:t.groupMate||'',plateConfidence:Number(t.plateConfidence||0),plate2Confidence:Number(t.plate2Confidence||0),score:Number(t.score||0),hits:Number(t.hits||0),anprAttempts:Number(t.anprAttempts||0)});
   }
   activeTracks=out;return out;
 }
 return parseIdentityRows();
}

function parseIdentityRows(){
 const rows=[...document.querySelectorAll('#identities .idrow')],out=new Map();for(const r of rows){const c=[...r.children].map(x=>x.textContent.trim());if(c.length<4)continue;const id=(c[0]||'').replace(/^#/,'');const type=c[1]||'';let plateCell=c[2]||'';let status=c[3]||'';plateCell=plateCell.replace(/^\?/,'');const parts=plateCell.split('/').map(normalizePlate).filter(Boolean);if(!id)continue;if(type==='Truck & Dog'&&parts[0]&&!parts[1]&&status==='CONFIRMED')status='PARTIAL';out.set(id,{id,type,plate:parts[0]||'',plate2:parts[1]||'',status})}activeTracks=out;return out;
}
async function frameQuality(){
 if(!video?.videoWidth)return null;const W=160,H=Math.max(90,Math.round(160*video.videoHeight/video.videoWidth)),c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(video,0,0,W,H);const d=x.getImageData(0,0,W,H).data;let sum=0,sum2=0,glare=0,dark=0,edge=0;const gray=new Float32Array(W*H);for(let i=0,j=0;i<gray.length;i++,j+=4){const g=.299*d[j]+.587*d[j+1]+.114*d[j+2];gray[i]=g;sum+=g;sum2+=g*g;if(g>242)glare++;if(g<28)dark++}for(let y=1;y<H-1;y++)for(let xx=1;xx<W-1;xx++){const i=y*W+xx;edge+=Math.abs(gray[i+1]-gray[i-1])+Math.abs(gray[i+W]-gray[i-W])}const n=gray.length,mean=sum/n,contrast=Math.sqrt(Math.max(0,sum2/n-mean*mean)),sharp=edge/((W-2)*(H-2)*2);return{brightness:mean/255,contrast:contrast/128,glare:glare/n,dark:dark/n,sharpness:sharp/32}}
async function hashFrame(){const c=document.createElement('canvas');c.width=9;c.height=8;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(video,0,0,9,8);const d=x.getImageData(0,0,9,8).data,bits=[];for(let y=0;y<8;y++)for(let xx=0;xx<8;xx++){const i=(y*9+xx)*4,j=(y*9+xx+1)*4;const a=d[i]+d[i+1]+d[i+2],b=d[j]+d[j+1]+d[j+2];bits.push(a>b?1:0)}let h='';for(let i=0;i<64;i+=4)h+=parseInt(bits.slice(i,i+4).join(''),2).toString(16);return h}
function ham(a,b){if(!a||!b||a.length!==b.length)return 64;let n=0;for(let i=0;i<a.length;i++){let v=parseInt(a[i],16)^parseInt(b[i],16);while(v){n+=v&1;v>>=1}}return n}
async function hashBlob(blob){try{const im=await createImageBitmap(blob),c=document.createElement('canvas');c.width=9;c.height=8;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(im,0,0,9,8);im.close?.();const d=x.getImageData(0,0,9,8).data,bits=[];for(let y=0;y<8;y++)for(let xx=0;xx<8;xx++){const i=(y*9+xx)*4,j=(y*9+xx+1)*4;bits.push((d[i]+d[i+1]+d[i+2])>(d[j]+d[j+1]+d[j+2])?1:0)}let h='';for(let i=0;i<64;i+=4)h+=parseInt(bits.slice(i,i+4).join(''),2).toString(16);return h}catch{return hashFrame()}}

async function captureTrackBlob(t){try{const b=await window.GateTrackerAPI?.crop?.(t.id);if(b)return b}catch{}return new Promise(res=>{if(!video?.videoWidth)return res(null);const max=1280,s=Math.min(1,max/video.videoWidth),w=Math.round(video.videoWidth*s),h=Math.round(video.videoHeight*s),c=document.createElement('canvas');c.width=w;c.height=h;c.getContext('2d').drawImage(video,0,0,w,h);c.toBlob(res,'image/jpeg',.9)})}
async function blobQuality(blob){if(!blob)return null;try{const im=await createImageBitmap(blob),W=160,H=Math.max(70,Math.round(160*im.height/im.width)),c=document.createElement('canvas');c.width=W;c.height=H;const x=c.getContext('2d',{willReadFrequently:true});x.drawImage(im,0,0,W,H);im.close?.();const d=x.getImageData(0,0,W,H).data;let sum=0,sum2=0,glare=0,dark=0,edge=0;const gray=new Float32Array(W*H);for(let i=0,j=0;i<gray.length;i++,j+=4){const g=.299*d[j]+.587*d[j+1]+.114*d[j+2];gray[i]=g;sum+=g;sum2+=g*g;if(g>242)glare++;if(g<28)dark++}for(let y=1;y<H-1;y++)for(let xx=1;xx<W-1;xx++){const i=y*W+xx;edge+=Math.abs(gray[i+1]-gray[i-1])+Math.abs(gray[i+W]-gray[i-W])}const n=gray.length,mean=sum/n,contrast=Math.sqrt(Math.max(0,sum2/n-mean*mean)),sharp=edge/((W-2)*(H-2)*2);return{brightness:mean/255,contrast:contrast/128,glare:glare/n,dark:dark/n,sharpness:sharp/32}}catch{return frameQuality()}}
function hardScore(q,t){let s=0,reasons=[];if(t.status==='PARTIAL'){s+=.42;reasons.push('partial_identity')}else if(t.status!=='CONFIRMED'){s+=.30;reasons.push('unresolved_identity')}if(t.type==='Truck & Dog'&&!t.plate2){s+=.16;reasons.push('missing_dog_plate')}if((t.plateConfidence||0)>0&&(t.plateConfidence||0)<.75){s+=.10;reasons.push('weak_plate_consensus')}if((t.score||1)<.48){s+=.08;reasons.push('weak_vehicle_detection')}if((t.anprAttempts||0)>=3&&!t.plate){s+=.10;reasons.push('repeated_anpr_failure')}if(t.phase==='GATE'){s+=.06;reasons.push('gate_transition')}s+=Math.min(.18,Math.max(0,(.45-q.sharpness)*.42));if(q.glare>.08){s+=Math.min(.16,q.glare*1.25);reasons.push('glare')}if(q.dark>.12){s+=Math.min(.16,q.dark*1.35);reasons.push('dark')}if(Math.abs(q.brightness-.5)>.28){s+=.05;reasons.push('exposure')}return{score:Math.min(1,s),reasons}}
async function storeActiveFrame(t,caseId,seq){
 const blob=await captureTrackBlob(t);if(!blob)return false;const q=await blobQuality(blob);if(!q)return false;const hash=await hashBlob(blob),h=hardScore(q,t),existing=(await all()).filter(x=>x.trackId===t.id);
 const near=existing.filter(x=>ham(x.hash,hash)<8).sort((a,b)=>(b.valueScore||b.hardScore||0)-(a.valueScore||a.hardScore||0))[0],value=Math.min(1,h.score+.08*Math.min(1,q.sharpness)+.04*Math.min(1,q.contrast));
 if(near){if((near.valueScore||near.hardScore||0)>=value-.02)return false;await del(near.id)}
 const current=(await all()).filter(x=>x.trackId===t.id);
 if(current.length>=9){const low=[...current].sort((a,b)=>(a.valueScore||a.hardScore||0)-(b.valueScore||b.hardScore||0))[0];if((low.valueScore||low.hardScore||0)>=value-.04)return false;await del(low.id)}
 const id=`${Date.now()}-${String(t.shortId||t.id)}-${seq}-${Math.random().toString(36).slice(2,6)}`;
 await put({id,caseId,sequenceIndex:seq,trackId:t.id,shortTrackId:t.shortId||t.id,groupId:t.groupId||null,groupMate:t.groupMate||null,type:t.type,status:t.status,phase:t.phase||null,candidate:t.plate||t.candidate||'',candidate2:t.plate2||t.candidate2||'',label:t.status==='CONFIRMED'?t.plate:'',label2:t.status==='CONFIRMED'?t.plate2:'',plateConfidence:t.plateConfidence||0,plate2Confidence:t.plate2Confidence||0,vehicleScore:t.score||0,hits:t.hits||0,anprAttempts:t.anprAttempts||0,quality:q,hardScore:h.score,valueScore:value,reasons:h.reasons,hash,createdAt:Date.now(),blob});
 return true;
}
async function captureBurst(t,hard){
 if(burstBusy.has(t.id))return;burstBusy.add(t.id);const caseId='case-'+Date.now()+'-'+String(t.shortId||t.id);
 try{
   await storeActiveFrame(t,caseId,0);
   if(hard>=.46||t.status==='PARTIAL'){
     for(let seq=1;seq<3;seq++){await new Promise(r=>setTimeout(r,360));const current=coreTracks().get(t.id);if(!current)break;await storeActiveFrame(current,caseId,seq)}
   }
 }finally{burstBusy.delete(t.id);lastCapture=Date.now();updateStats();if(specialistUrl&&t.status!=='CONFIRMED')queueAdvanced(t.id)}
}
async function maybeCapture(){
 if(Date.now()-lastCapture<1300||!db||!video?.videoWidth)return;
 const tracks=coreTracks();if(!tracks.size)return;
 const prelim=[...tracks.values()].map(t=>({t,p:(t.status==='PARTIAL'?.75:t.status!=='CONFIRMED'?.48:0)+(t.type==='Truck & Dog'&&!t.plate2?.18:0)+(t.phase==='GATE'?.08:0)+Math.min(.12,(t.anprAttempts||0)*.025)+(1-Math.min(1,t.score||.5))*.08})).sort((a,b)=>b.p-a.p).slice(0,2),ranked=[];
 for(const x of prelim){const blob=await captureTrackBlob(x.t);if(!blob)continue;const q=await blobQuality(blob);if(!q)continue;const h=hardScore(q,x.t);if(h.score<.24&&x.t.status==='CONFIRMED')continue;ranked.push({t:x.t,h:h.score})}
 if(!ranked.length)return;ranked.sort((a,b)=>b.h-a.h);captureBurst(ranked[0].t,ranked[0].h).catch(()=>{});
}
async function propagateLabels(){const tracks=coreTracks();const samples=await all();let changed=0;for(const t of tracks.values()){if(!t.plate)continue;for(const s of samples){let dirty=false;if(s.trackId===t.id&&!s.label){s.label=t.plate;s.labelSource='tracker-confirmed';dirty=true}if(s.trackId===t.id&&t.plate2&&!s.label2){s.label2=t.plate2;s.label2Source='tracker-confirmed';dirty=true}if(dirty){await put(s);changed++}}}if(changed)updateStats()}
async function updateStats(){if(!db)return;const a=await all(),cases=new Set(a.map(x=>x.caseId||x.id));$('caseCount').textContent=cases.size;$('labelCount').textContent=a.filter(x=>x.label).length+' labelled';$('hardCount').textContent=a.filter(x=>(x.valueScore||x.hardScore||0)>=.55).length+' review';$('learnState').textContent=specialistUrl?'AU local + advanced hard cases':'AU local active learning'}
async function labelLatest(){const a=(await all()).filter(x=>!x.label||(x.type==='Truck & Dog'&&!x.label2)).sort((x,y)=>y.createdAt-x.createdAt);if(!a.length){$('learnHint').textContent='No unlabelled hard cases.';return}const target=a[0],v=prompt(`Primary/prime plate for track #${target.trackId}:`,target.label||target.candidate||'');if(v===null)return;const p=normalizePlate(v);if(p.length<3)return;let p2=target.label2||target.candidate2||'';if(target.type==='Truck & Dog'){const v2=prompt('Dog/trailer plate (leave blank if not readable):',p2);if(v2!==null)p2=normalizePlate(v2)}const same=(await all()).filter(x=>x.trackId===target.trackId);for(const s of same){s.label=p;s.labelSource='manual';if(p2){s.label2=p2;s.label2Source='manual'}await put(s)}$('learnHint').textContent=`Track #${target.trackId} labelled ${p}${p2?' / '+p2:''}.`;updateStats()}
function setSpecialist(){const v=prompt('Advanced recogniser URL (e.g. http://192.168.1.20:8787). Leave blank to disable.',specialistUrl);if(v===null)return;specialistUrl=v.trim().replace(/\/$/,'');localStorage.setItem('gateSpecialistUrl',specialistUrl);$('learnHint').textContent=specialistUrl?'Advanced multi-frame recogniser enabled.':'Advanced recogniser disabled; local learning still active.';updateStats()}
async function queueAdvanced(trackId){if(advancedBusy||!specialistUrl)return;const a=(await all()).filter(x=>x.trackId===trackId).sort((x,y)=>(y.valueScore||y.hardScore||0)-(x.valueScore||x.hardScore||0)||y.createdAt-x.createdAt).slice(0,8);if(a.length<3)return;advancedBusy=true;try{const fd=new FormData();for(const s of a)fd.append('frames',s.blob,`${s.id}.jpg`);fd.append('qualities',JSON.stringify(a.map(s=>s.quality)));const r=await fetch(specialistUrl+'/recognize',{method:'POST',body:fd});if(!r.ok)throw new Error('HTTP '+r.status);const j=await r.json(),plate=normalizePlate(j.text||'');if(plate){for(const s of a){s.advanced={text:plate,confidence:j.confidence,entropy:j.entropy,status:j.status};if(!s.label&&j.status==='CONFIRMED'&&Number(j.confidence)>=.85){s.label=plate;s.labelSource='advanced-high-confidence'}await put(s)}$('learnHint').textContent=`Advanced: ${plate} · ${Math.round((j.confidence||0)*100)}% · ${j.status||''}`}}catch(e){$('learnHint').textContent='Advanced recogniser unavailable; local tracking/learning continues.'}finally{advancedBusy=false;updateStats()}}
async function exportZip(){const a=await all();if(!a.length)return alert('No learning samples yet.');const m=await import('https://cdn.jsdelivr.net/npm/jszip@3.10.1/+esm'),JSZip=m.default||m,zip=new JSZip(),manifest=[];for(const s of a){const fn=`images/${s.id}.jpg`;zip.file(fn,s.blob);manifest.push({image:fn,case_id:s.caseId||null,sequence_index:s.sequenceIndex??null,track_id:s.trackId,short_track_id:s.shortTrackId||null,group_id:s.groupId||null,group_mate:s.groupMate||null,vehicle_type:s.type,status:s.status,phase:s.phase||null,candidate:s.candidate,candidate2:s.candidate2||null,label:s.label||null,label2:s.label2||null,label_source:s.labelSource||null,label2_source:s.label2Source||null,plate_confidence:s.plateConfidence??null,secondary_plate_confidence:s.plate2Confidence??null,vehicle_score:s.vehicleScore??null,hits:s.hits??null,anpr_attempts:s.anprAttempts??null,quality:s.quality,hard_score:s.hardScore,value_score:s.valueScore??s.hardScore,reasons:s.reasons||[],hash:s.hash,created_at:new Date(s.createdAt).toISOString(),advanced:s.advanced||null})}zip.file('manifest.jsonl',manifest.map(x=>JSON.stringify(x)).join('\n'));zip.file('README.txt','Gate Specialist v8 active-learning dataset. Only diverse/high-value cases are retained. case_id + sequence_index group multi-frame bursts. Primary and secondary labels represent prime-mover and dog/trailer plates when available. manifest.jsonl includes uncertainty reasons, image quality, identity, operational state and advanced-recogniser results.');const blob=await zip.generateAsync({type:'blob',compression:'DEFLATE',compressionOptions:{level:6}}),u=URL.createObjectURL(blob),link=document.createElement('a');link.href=u;link.download=`gate-specialist-${new Date().toISOString().slice(0,10)}.zip`;link.click();setTimeout(()=>URL.revokeObjectURL(u),3000)}
async function clearAll(){if(!confirm('Clear all locally stored gate-specialist training samples?'))return;const a=await all();for(const s of a)await del(s.id);updateStats()}

await openDB();injectUI();updateStats();try{await navigator.storage?.persist?.()}catch{}setInterval(()=>maybeCapture().catch(()=>{}),900);setInterval(()=>propagateLabels().catch(()=>{}),1200);
