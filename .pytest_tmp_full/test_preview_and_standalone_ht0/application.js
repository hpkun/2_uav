// APP_JS_START
'use strict';
const P = PAYLOAD;
const ids = P.entity_ids, n = P.time_s.length;
let logicalFrame = 0, renderedFrame = -1;
let playing = false, cameraInteracting = false, renderBusy = false, renderPending = false;
let playbackRaf = null;
let playAnchorVisualTime = 0, playAnchorWallTime = 0, speed = 1, trailSeconds = -1;
let wheelEndTimer = null, wheelActive = false, pointerActive = false, currentView = 'episode';
const graph = document.getElementById('graph'), loading = document.getElementById('loading');
const idx = {trajectory: [], current: [], heading: [], deaths: 3*ids.length, attacks: 3*ids.length+1, ground: 3*ids.length+2};
for (let i=0;i<ids.length;i++) { idx.trajectory.push(i); idx.current.push(ids.length+i); idx.heading.push(2*ids.length+i); }
const options = id => document.getElementById(id);
function esc(x) { return String(x).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function eventText(e) {
  if (e.type === 'attack') return `${e.attacker} → ${e.target} ATTACK`;
  if (e.type === 'red_separation_warning') return `RED SEPARATION WARNING (${Number(e.minimum_distance).toFixed(0)} m)`;
  const labels = {boundary:'BOUNDARY LOSS', blue_escape:'BLUE ESCAPE', red_attack:'DESTROYED [red_attack]', blue_attack:'DESTROYED [blue_attack]'};
  return `${e.entity} ${labels[e.cause] || `LOST [${e.cause}]`}`;
}
function frameAtOrBeforeTime(time) {
  if (time <= P.time_s[0]) return 0;
  if (time >= P.time_s[n-1]) return n-1;
  let low=0, high=n-1;
  while (low <= high) {
    const mid=(low+high)>>1;
    if (P.time_s[mid] <= time) low=mid+1; else high=mid-1;
  }
  return Math.max(0, high);
}
function updatePanels(frame) {
  const t=P.time_s[frame];
  options('clock').textContent = `Visual Time: ${t.toFixed(1)} s | Raw Decision Step: ${P.raw_step[frame]}`;
  options('state').innerHTML = ids.map((id,i) => `<div class="state-row ${id==='MAV'?'mav':''}"><span>${id}</span><span>${P.alive[frame][i]?'Alive':'Lost'} · ${P.kinematics[frame][i][3].toFixed(1)} m/s</span></div>`).join('');
  const recent = P.events.filter(e => Number(e.time_s) <= t + 1e-9).slice(-5).reverse();
  options('events').innerHTML = recent.length ? recent.map(e => `<div>${Number(e.time_s).toFixed(1)} s&nbsp; ${esc(eventText(e))}</div>`).join('') : '<div class="muted">No events yet</div>';
  options('result').style.display = frame === n-1 ? 'block' : 'none';
  options('slider').value=frame;
}
function trajectory(i, start, frame) {
  const x=[],y=[],z=[];
  for(let f=start;f<=frame;f++) if(P.alive[f][i]) { const q=P.kinematics[f][i]; x.push(q[0]/1000);y.push(q[1]/1000);z.push(q[2]/1000); }
  return {x,y,z};
}
function hover(i,q,alive) {
  const id=ids[i], type=P.entity_types[id], spec=P.aircraft_specs[type];
  return `Agent: ${id}<br>Team: ${P.entity_teams[id]}<br>Type: ${type}<br>X: ${(q[0]/1000).toFixed(3)} km<br>Y: ${(q[1]/1000).toFixed(3)} km<br>Altitude: ${(q[2]/1000).toFixed(3)} km<br>Speed: ${q[3].toFixed(1)} m/s<br>Allowed Speed: ${spec.v_min}–${spec.v_max} m/s<br>Heading: ${(q[5]*180/Math.PI).toFixed(1)}°<br>Pitch: ${(q[4]*180/Math.PI).toFixed(1)}°<br>Alive: ${alive}`;
}
async function renderFrame(frameIndex) {
  const f=frameIndex, t=P.time_s[f];
  const start=trailSeconds<0?0:frameAtOrBeforeTime(Math.max(P.time_s[0],t-trailSeconds));
  const tx=[],ty=[],tz=[],tv=[];
  const cx=[],cy=[],cz=[],ct=[],ch=[],cv=[];
  const hx=[],hy=[],hz=[],hv=[];
  for(let i=0;i<ids.length;i++) {
    const tr=trajectory(i,start,f), q=P.kinematics[f][i], live=P.alive[f][i];
    tx.push(tr.x);ty.push(tr.y);tz.push(tr.z);tv.push(true);
    const text=hover(i,q,live), label=options('showLabels').checked?ids[i]:'';
    cx.push(live?[q[0]/1000]:[]);cy.push(live?[q[1]/1000]:[]);cz.push(live?[q[2]/1000]:[]);
    ct.push([label]);ch.push([text]);cv.push(live);
    let x=[],y=[],z=[];
    if(live) { const L=.5,th=q[4],ps=q[5]; x=[q[0]/1000,q[0]/1000+L*Math.cos(th)*Math.cos(ps)]; y=[q[1]/1000,q[1]/1000+L*Math.cos(th)*Math.sin(ps)]; z=[q[2]/1000,q[2]/1000+L*Math.sin(th)]; }
    hx.push(x);hy.push(y);hz.push(z);hv.push(live && options('showHeadings').checked);
  }
  await Plotly.restyle(graph,{x:tx,y:ty,z:tz,visible:tv},idx.trajectory);
  await Plotly.restyle(graph,{x:cx,y:cy,z:cz,text:ct,hovertext:ch,visible:cv},idx.current);
  await Plotly.restyle(graph,{x:hx,y:hy,z:hz,visible:hv},idx.heading);
  const dx=[],dy=[],dz=[],dt=[];
  if(options('showDeaths').checked) for(const d of P.deaths) if(Number(d.time_s)<=t+1e-9) { dx.push(d.position[0]/1000);dy.push(d.position[1]/1000);dz.push(d.position[2]/1000);dt.push(`${d.entity}<br>Type: ${P.entity_types[d.entity]}<br>Death Time: ${Number(d.time_s).toFixed(1)} s<br>Cause: ${d.cause}<br>X/Y/Altitude: ${(d.position[0]/1000).toFixed(3)} / ${(d.position[1]/1000).toFixed(3)} / ${(d.position[2]/1000).toFixed(3)} km`); }
  await Plotly.restyle(graph,{x:[dx],y:[dy],z:[dz],hovertext:[dt],visible:options('showDeaths').checked},[idx.deaths]);
  const ax=[],ay=[],az=[];
  if(options('showAttacks').checked) for(const e of P.events) if(e.type==='attack' && t-Number(e.time_s)>=-1e-9 && t-Number(e.time_s)<=.8+1e-9) { const f=e.trace_frame,a=P.raw_kinematics[f][ids.indexOf(e.attacker)],b=P.raw_kinematics[f][ids.indexOf(e.target)]; ax.push(a[0]/1000,b[0]/1000,null);ay.push(a[1]/1000,b[1]/1000,null);az.push(a[2]/1000,b[2]/1000,null); }
  await Plotly.restyle(graph,{x:[ax],y:[ay],z:[az],visible:options('showAttacks').checked},[idx.attacks]);
  return true;
}
async function runCombatMutationScheduler() {
  if(renderBusy)return;
  renderBusy=true;let failed=false;
  try {
    while(true) {
      if(cameraInteracting)break;
      if(!renderPending && renderedFrame===logicalFrame)break;
      renderPending=false;
      const target=logicalFrame, completed=await renderFrame(target);
      if(completed)renderedFrame=target;else renderPending=true;
    }
  } catch(error) {
    console.error(error);renderPending=true;failed=true;
  } finally {
    renderBusy=false;
    if(!failed && !cameraInteracting && (renderPending || renderedFrame!==logicalFrame)) {
      queueMicrotask(runCombatMutationScheduler);
    }
  }
}
function requestRender() {
  renderPending=true;
  if(cameraInteracting || renderBusy)return;
  runCombatMutationScheduler();
}
function logicalVisualTime(now) {
  if(!playing) return Math.min(P.time_s[n-1],playAnchorVisualTime);
  return Math.min(P.time_s[n-1],playAnchorVisualTime+(now-playAnchorWallTime)/1000*speed);
}
function playbackLoop(now) {
  if(!playing) return;
  const targetTime=logicalVisualTime(now), nextFrame=frameAtOrBeforeTime(targetTime);
  if(nextFrame!==logicalFrame) {logicalFrame=nextFrame;if(!cameraInteracting)updatePanels(logicalFrame);}
  if(cameraInteracting) renderPending=true; else if(logicalFrame!==renderedFrame) requestRender();
  if(targetTime>=P.time_s[n-1]) {
    playing=false;playbackRaf=null;playAnchorVisualTime=P.time_s[n-1];logicalFrame=n-1;
    if(cameraInteracting)renderPending=true;else {updatePanels(logicalFrame);requestRender();}return;
  }
  playbackRaf=requestAnimationFrame(playbackLoop);
}
function stopPlaybackClock() {
  const now=performance.now();
  if(playing) {playAnchorVisualTime=logicalVisualTime(now);logicalFrame=frameAtOrBeforeTime(playAnchorVisualTime);}
  playing=false;if(playbackRaf!==null){cancelAnimationFrame(playbackRaf);playbackRaf=null;}
}
function pause() {
  stopPlaybackClock();if(!cameraInteracting)updatePanels(logicalFrame);
  if(cameraInteracting)renderPending=true;else if(logicalFrame!==renderedFrame)requestRender();
}
function play() {
  if(playing || logicalFrame>=n-1)return;
  playing=true;playAnchorWallTime=performance.now();playbackRaf=requestAnimationFrame(playbackLoop);
}
function setFrame(frame) {
  stopPlaybackClock();logicalFrame=Math.max(0,Math.min(n-1,Number(frame)));playAnchorVisualTime=P.time_s[logicalFrame];
  updatePanels(logicalFrame);if(cameraInteracting)renderPending=true;else requestRender();
}
function beginCameraInteraction() {
  if(cameraInteracting)return;
  cameraInteracting=true;
}
function endCameraInteraction() {
  if(!cameraInteracting || pointerActive || wheelActive)return;
  cameraInteracting=false;
  if(renderedFrame!==logicalFrame) {updatePanels(logicalFrame);renderPending=true;requestRender();}
  else renderPending=false;
}
async function setView(name,ranges,showGround){
  currentView=name;
  await Plotly.restyle(graph,{visible:showGround,x:[[ranges.x[0],ranges.x[1],ranges.x[1],ranges.x[0]]],y:[[ranges.y[0],ranges.y[0],ranges.y[1],ranges.y[1]]],z:[[0,0,0,0]]},[idx.ground]);
  return Plotly.relayout(graph,{
    'scene.xaxis.autorange':false,'scene.yaxis.autorange':false,'scene.zaxis.autorange':false,
    'scene.xaxis.range':ranges.x,'scene.yaxis.range':ranges.y,'scene.zaxis.range':ranges.z,
    'scene.aspectmode':'cube'
  });
}
function setEpisodeView(){return setView('episode',P.episode_ranges,true);}
function setFullBattlefieldView(){return setView('full',P.full_ranges,false);}
async function init(){
 try {
  if(typeof Plotly==='undefined') throw new Error('Embedded Plotly library is unavailable.');
  const data=[];
  for(let i=0;i<ids.length;i++) data.push({type:'scatter3d',mode:'lines',name:ids[i],x:[],y:[],z:[],line:{color:P.styles[ids[i]].color,width:P.styles[ids[i]].width,dash:P.styles[ids[i]].dash},hoverinfo:'skip'});
  for(let i=0;i<ids.length;i++) data.push({type:'scatter3d',mode:'markers+text',showlegend:false,x:[],y:[],z:[],text:[],textposition:'top center',hoverinfo:'text',marker:{color:P.styles[ids[i]].color,size:ids[i]==='MAV'?8:6,symbol:P.styles[ids[i]].marker,line:{color:'#fff',width:1}}});
  for(let i=0;i<ids.length;i++) data.push({type:'scatter3d',mode:'lines',showlegend:false,x:[],y:[],z:[],hoverinfo:'skip',line:{color:P.styles[ids[i]].color,width:4}});
  data.push({type:'scatter3d',mode:'markers',name:'Loss',x:[],y:[],z:[],hoverinfo:'text',marker:{symbol:'x',size:8,color:'#222'}});
  data.push({type:'scatter3d',mode:'lines',name:'Attack',x:[],y:[],z:[],hoverinfo:'skip',line:{color:'#ed3b3b',width:7}});
  data.push({type:'mesh3d',name:'Ground reference',showlegend:false,hoverinfo:'skip',x:[P.episode_ranges.x[0],P.episode_ranges.x[1],P.episode_ranges.x[1],P.episode_ranges.x[0]],y:[P.episode_ranges.y[0],P.episode_ranges.y[0],P.episode_ranges.y[1],P.episode_ranges.y[1]],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:'#8d99a6',opacity:.07,flatshading:true});
  const layout={paper_bgcolor:'#f5f7fa',plot_bgcolor:'#fff',margin:{l:0,r:0,t:35,b:0},legend:{x:.01,y:.99,bgcolor:'rgba(255,255,255,.8)'},uirevision:'combat-camera-v1',scene:{xaxis:{title:'X / km',range:P.episode_ranges.x,autorange:false},yaxis:{title:'Y / km',range:P.episode_ranges.y,autorange:false},zaxis:{title:'Altitude / km',range:P.episode_ranges.z,autorange:false},aspectmode:'cube',dragmode:'orbit',camera:P.default_camera}};
  await Plotly.newPlot(graph,data,layout,{responsive:true,displaylogo:false,scrollZoom:true});
  if(await renderFrame(0))renderedFrame=0;updatePanels(0);loading.style.display='none';
 } catch(error) { console.error(error);loading.className='error';loading.innerHTML=`<b>Interactive replay initialization failed</b><br>${esc(error.message)}<br>Check browser WebGL support.`; }
}
options('play').onclick=play;options('pause').onclick=pause;options('restart').onclick=()=>setFrame(0);
options('prev').onclick=()=>setFrame(logicalFrame-1);options('next').onclick=()=>setFrame(logicalFrame+1);
options('slider').oninput=e=>setFrame(e.target.value);
options('speed').onchange=e=>{const now=performance.now();if(playing){playAnchorVisualTime=logicalVisualTime(now);playAnchorWallTime=now;}speed=Number(e.target.value);};
options('trail').onchange=e=>{trailSeconds=Number(e.target.value);renderPending=true;if(!cameraInteracting)requestRender();};
for(const id of ['showHeadings','showLabels','showDeaths','showAttacks']) options(id).onchange=()=>{renderPending=true;if(!cameraInteracting)requestRender();};
options('resetCamera').onclick=()=>Plotly.relayout(graph,{'scene.camera':P.default_camera});options('episodeView').onclick=setEpisodeView;options('fullView').onclick=setFullBattlefieldView;
graph.addEventListener('pointerdown',()=>{pointerActive=true;beginCameraInteraction();},true);
window.addEventListener('pointerup',()=>{pointerActive=false;endCameraInteraction();},true);
window.addEventListener('pointercancel',()=>{pointerActive=false;endCameraInteraction();},true);
graph.addEventListener('wheel',()=>{wheelActive=true;beginCameraInteraction();if(wheelEndTimer!==null)clearTimeout(wheelEndTimer);wheelEndTimer=setTimeout(()=>{wheelActive=false;wheelEndTimer=null;endCameraInteraction();},200);},{capture:true,passive:true});
init();
