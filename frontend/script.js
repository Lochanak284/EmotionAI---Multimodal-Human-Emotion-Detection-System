const API = 'http://127.0.0.1:8000/predict';
const EMOTIONS = ['angry','disgust','fear','happy','neutral','sad','surprise'];
const COLORS = {
  angry:'#f43f5e', disgust:'#10b981', fear:'#8b5cf6',
  happy:'#f59e0b', neutral:'#64748b', sad:'#3b82f6', surprise:'#ec4899'
};

function switchTab(t) {
  document.querySelectorAll('.tab').forEach((b,i) =>
    b.classList.toggle('active', (i===0&&t==='live')||(i===1&&t==='upload'))
  );
  document.getElementById('panel-live').classList.toggle('active', t==='live');
  document.getElementById('panel-upload').classList.toggle('active', t==='upload');
}

let stream = null, camActive = false;

async function toggleCamera() {
  if (!camActive) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user', width:320, height:240 } });
      const v = document.getElementById('camVideo');
      v.srcObject = stream; v.style.display = 'block';
      document.getElementById('camEmpty').style.display = 'none';
      document.getElementById('camBtn').textContent = '⏹ Stop Camera';
      document.getElementById('camBtn').className = 'ctrl active-ctrl';
      setStatus('camStatus','ON','on');
      document.getElementById('camCard').classList.add('lit');
      camActive = true;
    } catch { alert('Camera access denied.'); }
  } else { stopCamera(); }
}

function stopCamera() {
  if (stream) { stream.getTracks().forEach(t=>t.stop()); stream=null; }
  document.getElementById('camVideo').style.display='none';
  document.getElementById('camVideo').srcObject=null;
  document.getElementById('camEmpty').style.display='flex';
  document.getElementById('camBtn').textContent='▶ Start Camera';
  document.getElementById('camBtn').className='ctrl';
  setStatus('camStatus','OFF','');
  document.getElementById('camCard').classList.remove('lit');
  camActive=false;
}

function captureFrame() {
  if (!camActive) return null;
  const v=document.getElementById('camVideo'), c=document.getElementById('captureCanvas');
  c.width=v.videoWidth||320; c.height=v.videoHeight||240;
  c.getContext('2d').drawImage(v,0,0);
  return new Promise(res=>c.toBlob(res,'image/jpeg',0.85));
}

let mediaRecorder=null, audioChunks=[], recActive=false, recTimer=null, recSecs=0;
let audioBlob=null, analyser=null, audioCtx=null, vizId=null;

async function toggleRecording() {
  if (!recActive) {
    try {
      const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new (window.AudioContext||window.webkitAudioContext)();
      const src = audioCtx.createMediaStreamSource(mic);
      analyser = audioCtx.createAnalyser(); analyser.fftSize=256;
      src.connect(analyser); drawWave();

      const preferred = ['audio/ogg;codecs=opus','audio/ogg','audio/webm'];
      const mimeType = preferred.find(m=>MediaRecorder.isTypeSupported(m)) || '';
      mediaRecorder = new MediaRecorder(mic, mimeType ? {mimeType} : {});

      audioChunks=[];
      mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
      mediaRecorder.onstop = () => {
        audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
        mic.getTracks().forEach(t=>t.stop());
        document.getElementById('voiceHint').textContent = `✅ ${recSecs}s recording ready`;
        cancelAnimationFrame(vizId); drawFlat();
      };

      mediaRecorder.start();
      startCaptions();
      recActive=true; recSecs=0;
      document.getElementById('voiceBtn').textContent='⏹ Stop Recording';
      document.getElementById('voiceBtn').className='ctrl rec-ctrl';
      setStatus('voiceStatus','REC','rec');
      document.getElementById('voiceCard').classList.add('lit');

      recTimer = setInterval(()=>{
        recSecs++;
        const m=Math.floor(recSecs/60), s=recSecs%60;
        document.getElementById('voiceTimer').textContent=`${m}:${String(s).padStart(2,'0')}`;
        if (recSecs>=20) stopRecording();
      },1000);
    } catch { alert('Microphone access denied.'); }
  } else { stopRecording(); }
}

function stopRecording() {
  stopCaptions();
  if (mediaRecorder && mediaRecorder.state!=='inactive') mediaRecorder.stop();
  clearInterval(recTimer); recActive=false;
  document.getElementById('voiceBtn').textContent='🎙️ Record Again';
  document.getElementById('voiceBtn').className='ctrl active-ctrl';
  setStatus('voiceStatus','DONE','on');
}

function drawWave() {
  const cv=document.getElementById('voiceCanvas'), ctx=cv.getContext('2d');
  cv.width=cv.offsetWidth; cv.height=cv.offsetHeight;
  const buf=new Uint8Array(analyser.frequencyBinCount);
  function frame() {
    vizId=requestAnimationFrame(frame);
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.strokeStyle='#8b5cf6'; ctx.lineWidth=2;
    ctx.beginPath();
    const sw=cv.width/buf.length; let x=0;
    buf.forEach((v,i)=>{ const y=(v/128)*(cv.height/2); i===0?ctx.moveTo(x,y):ctx.lineTo(x,y); x+=sw; });
    ctx.stroke();
  }
  frame();
}

function drawFlat() {
  const cv=document.getElementById('voiceCanvas'), ctx=cv.getContext('2d');
  cv.width=cv.offsetWidth; cv.height=cv.offsetHeight;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#06b6d4'; ctx.lineWidth=2;
  ctx.beginPath(); ctx.moveTo(0,cv.height/2); ctx.lineTo(cv.width,cv.height/2); ctx.stroke();
}

let recognition=null, capFinal='';

function startCaptions() {
  const box=document.getElementById('captionBox');
  box.style.display='block';
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  if (!SR) { document.getElementById('captionText').textContent='⚠️ Captions need Chrome/Edge.'; return; }
  capFinal=''; recognition=new SR();
  recognition.continuous=true; recognition.interimResults=true; recognition.lang='en-US';
  box.classList.add('on');
  document.getElementById('captionText').textContent='';
  document.getElementById('captionLive').textContent='🎙 Listening…';
  recognition.onresult=e=>{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      const t=e.results[i][0].transcript;
      e.results[i].isFinal ? (capFinal+=t+' ') : (interim+=t);
    }
    document.getElementById('captionText').textContent=capFinal;
    document.getElementById('captionLive').textContent=interim?`⏳ ${interim}`:'';
  };
  recognition.onerror=e=>{ if(e.error!=='no-speech') document.getElementById('captionLive').textContent=`⚠️ ${e.error}`; };
  recognition.onend=()=>{
    if(recActive&&recognition){ try{recognition.start();}catch{} }
    else {
      box.classList.remove('on');
      document.getElementById('captionLive').textContent='';
      if(!capFinal.trim()) document.getElementById('captionLive').textContent='— no speech detected —';
    }
  };
  try{recognition.start();}catch{}
}

function stopCaptions(){
  if(recognition){ recognition.onend=null; recognition.stop(); recognition=null; }
  document.getElementById('captionBox').classList.remove('on');
  document.getElementById('captionLive').textContent='';
  if(capFinal.trim()) document.getElementById('captionText').textContent=capFinal.trim();
  else document.getElementById('captionLive').textContent='— no speech detected —';
}

let textTimer=null;

function onTextInput(){
  const txt=document.getElementById('liveText').value;
  document.getElementById('charCount').textContent=txt.length;
  const on=txt.trim().length>0;
  setStatus('textStatus',on?'ON':'OFF',on?'on':'');
  document.getElementById('textCard').classList.toggle('lit',on);
  clearTimeout(textTimer);
  if(txt.trim().length>=10){
    document.getElementById('autoPill').style.display='inline-flex';
    textTimer=setTimeout(()=>{ document.getElementById('autoPill').style.display='none'; analyzeLive(true); },2000);
  } else { document.getElementById('autoPill').style.display='none'; }
}

async function analyzeLive(silent=false){
  const text=document.getElementById('liveText').value.trim();
  const hasText=text.length>=1, hasVoice=!!audioBlob, hasCam=camActive;
  if(!hasText&&!hasVoice&&!hasCam){ if(!silent) alert('Enable at least one input.'); return; }
  showLoader();
  const form=new FormData();
  if(hasText) form.append('text',text);
  if(hasVoice) form.append('voice', audioBlob, 'recording.ogg');
  if(hasCam){ const b=await captureFrame(); if(b) form.append('face',b,'frame.jpg'); }
  try{
    const r=await fetch(API,{method:'POST',body:form});
    const data = await r.json();
    renderResult(data,{face:hasCam,voice:hasVoice,text:hasText});
  }catch(e){ if(!silent) alert('Cannot reach backend at '+API); }
  finally { hideLoader(); }
}

async function analyzeUpload(){
  const ff=document.getElementById('upFace').files[0];
  const vf=document.getElementById('upVoice').files[0];
  const txt=document.getElementById('upText').value.trim();
  if(!ff&&!vf&&!txt){ alert('Provide at least one input.'); return; }
  showLoader();
  const form=new FormData();
  if(ff) form.append('face',ff);
  if(vf) form.append('voice',vf);
  if(txt) form.append('text',txt);
  try{
    const r=await fetch(API,{method:'POST',body:form});
    const data = await r.json();
    renderResult(data,{face:!!ff,voice:!!vf,text:!!txt});
  }catch{ alert('Cannot reach backend at '+API); }
  finally{ hideLoader(); }
}

function renderResult(data, used){
  // Primary emotion
  document.getElementById('rEmoji').textContent=data.emoji;
  document.getElementById('rEmotion').textContent=data.emotion.toUpperCase();
  document.getElementById('rConf').textContent=(data.confidence*100).toFixed(1)+'%';
  const usedList=Object.entries(used).filter(([,v])=>v).map(([k])=>k).join(' + ');
  document.getElementById('rInputs').textContent='Inputs used: '+usedList;

  const probs=data.probabilities;
  const sorted=EMOTIONS.slice().sort((a,b)=>probs[b]-probs[a]);
  document.getElementById('barsDiv').innerHTML=sorted.map((e,i)=>`
    <div class="bar-row">
      <div class="bar-lbl ${i===0?'top':''}">${e}</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${((probs[e]||0)*100).toFixed(1)}%;background:${COLORS[e]}"></div>
      </div>
      <div class="bar-pct">${((probs[e]||0)*100).toFixed(1)}%</div>
    </div>`).join('');

  const totalConfidence = ['face','voice','text'].reduce((sum, key) => {
    return sum + (data.modalities[key] ? data.modalities[key].confidence : 0);
  }, 0);

  document.getElementById('modGrid').innerHTML=
    [['📷','face','FACE'],['🎙️','voice','VOICE'],['💬','text','TEXT']].map(([icon,key,label])=>{
      const m=data.modalities[key];
      const contribution = m ? ((m.confidence / totalConfidence) * 100).toFixed(1) : 0;
      return m
        ? `<div class="mod-card used">
            <div class="mod-icon">${icon}</div>
            <div class="mod-name">${label}</div>
            <div class="mod-emo" style="color:${COLORS[m.emotion]}; font-size:1.6em; margin:10px 0;">${contribution}%</div>
            <div class="mod-pct" style="color:var(--muted);">contribution</div>
           </div>`
        : `<div class="mod-card">
            <div class="mod-icon" style="opacity:0.25">${icon}</div>
            <div class="mod-name">${label}</div>
            <div class="mod-none">not used</div>
           </div>`;
    }).join('');


  renderGauge(
    data.stress_score,
    'stressArc', 'stressVal', 'stressLabel', 'stressDetail',
    getStressDetail(data.stress_score, data.emotion)
  );

  renderGauge(
    data.engagement_score,
    'engageArc', 'engageVal', 'engageLabel', 'engageDetail',
    getEngageDetail(data.engagement_score, data.emotion)
  );


  document.getElementById('results').style.display='block';
  document.getElementById('results').scrollIntoView({behavior:'smooth',block:'start'});

  renderEmpathyResponse(data.empathy_response || '');
}

function renderGauge(score, arcId, valId, labelId, detailId, detailText){
  const pct = Math.round(score * 100);
  const arcLen = 173;
  const filled = (score * arcLen).toFixed(1);

  document.getElementById(arcId).setAttribute('stroke-dasharray', `${filled} ${arcLen}`);
  document.getElementById(valId).textContent = pct + '%';

  let label, color;
  if(pct >= 75)      { label='HIGH';   color='#f43f5e'; }
  else if(pct >= 45) { label='MEDIUM'; color='#f59e0b'; }
  else               { label='LOW';    color='#10b981'; }

  document.getElementById(labelId).textContent = label;
  document.getElementById(labelId).style.color = color;
  document.getElementById(valId).style.color = color;
  document.getElementById(detailId).textContent = detailText;
}

function getStressDetail(score, emotion){
  if(score >= 0.75) return `High stress detected. Primary emotion "${emotion}" is strongly associated with psychological pressure.`;
  if(score >= 0.45) return `Moderate stress indicators present. Some tension detected in the "${emotion}" response.`;
  return `Low stress levels. The "${emotion}" state suggests a relatively calm emotional baseline.`;
}

function getEngageDetail(score, emotion){
  if(score >= 0.75) return `High engagement — strong emotional activation detected. The "${emotion}" response shows active involvement.`;
  if(score >= 0.45) return `Moderate engagement. Some emotional presence in the "${emotion}" state.`;
  return `Low engagement detected. The "${emotion}" state suggests emotional disengagement or passivity.`;
}

function renderEmpathyResponse(text) {
  document.getElementById('empathyLoading').style.display = 'none';

  if (!text || text.trim() === '') {
    document.getElementById('empathyContent').innerHTML =
      '<span style="color:var(--hint);font-size:0.85em;">⚠️ Could not load empathy response.</span>';
    document.getElementById('empathyContent').style.display = 'block';
    return;
  }

  const tipMatch = text.match(/TIP:\s*(.+)/s);
  const mainText = text.replace(/TIP:.+/s, '').trim();
  const tipText  = tipMatch ? tipMatch[1].trim() : null;

  document.getElementById('empathyContent').innerHTML =
    `<div>${mainText}</div>` +
    (tipText ? `<div class="emp-tip">💡 ${tipText}</div>` : '');

  document.getElementById('empathyContent').style.display = 'block';
}

function setStatus(id,text,cls){
  const el=document.getElementById(id);
  el.textContent=text; el.className='status-dot'+(cls?' '+cls:'');
}

function showLoader(){
  document.getElementById('loader').style.display='block';
  document.getElementById('results').style.display='none';
  document.getElementById('liveBtn').disabled=true;
}

function hideLoader(){
  document.getElementById('loader').style.display='none';
  document.getElementById('liveBtn').disabled=false;
}

function previewFile(inp,lbl){
  if(inp.files[0]) document.getElementById(lbl).innerHTML=
    `<div class="uz-icon">✅</div><div>${inp.files[0].name}</div>`;
}

window.addEventListener('load',()=>setTimeout(drawFlat,120));