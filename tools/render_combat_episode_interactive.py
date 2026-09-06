"""Create a standalone offline Plotly 3D combat replay HTML."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plotly.offline import get_plotlyjs

from tools.combat_visualization import (ENTITY_IDS, STYLES, death_records, episode_cube_ranges,
                                        interpolate_trace_for_visualization, load_trace)

APP_JS = r"""
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
// APP_JS_END
"""

HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Heterogeneous Combat Replay</title>
<style>html,body{margin:0;height:100%;font-family:Inter,Segoe UI,sans-serif;color:#172033;background:#f5f7fa}.top{padding:10px 16px;background:#172033;color:white}.top b{color:#ffbf47}.controls{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:8px 14px;background:white;border-bottom:1px solid #d9dee8}button,select{padding:6px 9px}.slider{flex:1;min-width:220px}.main{display:grid;grid-template-columns:minmax(500px,1fr) 300px;height:calc(100% - 112px)}#graph{height:100%;min-height:480px}.side{padding:12px;overflow:auto;background:white;border-left:1px solid #d9dee8}.panel{border:1px solid #d9dee8;border-radius:7px;padding:10px;margin-bottom:10px}.panel h3{margin:0 0 8px;font-size:14px}.state-row{display:flex;justify-content:space-between;padding:4px}.state-row.mav{font-weight:700;color:#c5163a}.muted{color:#7a8497}.result{display:none;background:#edf7ee}.error{color:#9b1c1c;background:#fff0f0}#loading{position:absolute;z-index:5;padding:12px;background:#fff;border:1px solid #aaa;margin:20px}@media(max-width:850px){.main{grid-template-columns:1fr;height:auto}.side{border-left:0}.main #graph{height:65vh}}</style>
<script>PLOTLY_JS</script></head><body><div class="top"><b>METHOD</b> · Evaluation Profile: PROFILE · Blue Policy: BLUE_MODE · qualitative visualization only</div>
<div class="controls"><button id="play">Play</button><button id="pause">Pause</button><button id="prev">Previous Frame</button><button id="next">Next Frame</button><button id="restart">Restart</button><input id="slider" class="slider" type="range" min="0" max="LAST" value="0"><select id="speed"><option value=".25">0.25x</option><option value=".5">0.5x</option><option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option></select><select id="trail"><option value="-1">Trail: Full</option><option value="5">Trail: 5s</option><option value="10">Trail: 10s</option><option value="20">Trail: 20s</option></select><label><input id="showHeadings" type="checkbox" checked> Headings</label><label><input id="showLabels" type="checkbox" checked> Labels</label><label><input id="showDeaths" type="checkbox" checked> Death markers</label><label><input id="showAttacks" type="checkbox" checked> Attack lines</label><button id="resetCamera">Reset Camera</button><button id="episodeView">Episode View</button><button id="fullView">Full Battlefield</button></div>
<div class="main"><div style="position:relative"><div id="loading">Loading interactive replay...</div><div id="graph"></div></div><aside class="side"><div class="panel"><h3 id="clock">Visual Time</h3></div><div class="panel"><h3>Current State</h3><div id="state"></div></div><div class="panel"><h3>Recent Events</h3><div id="events"></div></div><div class="panel result" id="result"><h3>RESULT_TITLE</h3><div>RESULT_BODY</div></div></aside></div>
<script>const PAYLOAD=PAYLOAD_JSON;</script><script>APPLICATION_JS</script></body></html>"""


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")


def render_interactive(input_dir: Path, output: Path | None = None, *, visual_dt: float = .1, overwrite: bool = False) -> Path:
    trace, meta = load_trace(input_dir); visual = interpolate_trace_for_visualization(trace, float(meta["decision_dt"]), visual_dt)
    directory = Path(input_dir).expanduser().resolve(); output = (output or directory / "episode_interactive.html").expanduser().resolve()
    if output.exists() and not overwrite: raise FileExistsError(f"refusing to overwrite existing render: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    field = meta["battlefield"]
    payload = {"entity_ids":list(ENTITY_IDS),"entity_types":meta["entity_types"],"entity_teams":meta["entity_teams"],
               "aircraft_specs":meta["aircraft_specs"],"styles":STYLES,"kinematics":visual["kinematics"].tolist(),
               "raw_kinematics":trace["kinematics"].tolist(),"alive":visual["alive"].tolist(),
               "time_s":visual["time_s"].tolist(),"raw_step":visual["raw_step"].tolist(),"visual_dt":visual_dt,
               "events":meta.get("events",[]),"deaths":death_records(trace,meta),
               "episode_ranges":episode_cube_ranges(trace["kinematics"],trace["alive"]),
               "full_ranges":{"x":[x/1000 for x in field["x"]],"y":[x/1000 for x in field["y"]],"z":[x/1000 for x in field["altitude"]]},
               "default_camera":{"eye":{"x":1.50,"y":-1.60,"z":1.25},"up":{"x":0,"y":0,"z":1}}}
    outcome={"red":"RED WIN","blue":"BLUE WIN","draw":"DRAW"}.get(meta["outcome"],str(meta["outcome"]).upper())
    body=(f"{'MAV SURVIVED' if meta['mav_survived'] else 'MAV LOST'}<br>UAV Survivors {meta['red_uav_survivors']}/3<br>"
          f"Blue Survivors {meta['blue_survivors']}/4<br>Red Attack Kills {meta['red_attack_kills']}<br>Blue Attack Kills {meta['blue_attack_kills']}<br>"
          f"Episode Return {meta['episode_return']:.3f}<br>Episode Length {meta['episode_length']}<br>Evaluation Profile {meta['evaluation_profile']}<br>Blue Policy {meta['blue_target_mode']}")
    html=HTML_TEMPLATE.replace("PLOTLY_JS",get_plotlyjs()).replace("PAYLOAD_JSON",_json(payload)).replace("APPLICATION_JS",APP_JS.replace("PAYLOAD","P"))
    # Undo the one placeholder reference: APP_JS expects global P assigned from PAYLOAD.
    html=html.replace("const P = P;","const P = PAYLOAD;").replace("METHOD",str(meta["algorithm"])).replace("PROFILE",str(meta["evaluation_profile"])).replace("BLUE_MODE",str(meta["blue_target_mode"])).replace("LAST",str(len(visual["time_s"])-1)).replace("RESULT_TITLE",outcome).replace("RESULT_BODY",body)
    output.write_text(html,encoding="utf-8"); return output


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input-dir",type=Path,required=True);p.add_argument("--output",type=Path);p.add_argument("--visual-dt",type=float,default=.1);p.add_argument("--overwrite",action="store_true");a=p.parse_args();print(render_interactive(a.input_dir,a.output,visual_dt=a.visual_dt,overwrite=a.overwrite))
if __name__ == "__main__":main()
