"""Standalone HTML application used by :mod:`uam4x.n2_viewer`.

The interaction vocabulary follows OpenMDAO's N2 viewer, while the model and
labels are native uam4x concepts.
"""

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--nav:#58b98b;--nav2:#318d68;--bg:#f5f7fa;--panel:#fff;--ink:#172033;
--muted:#667085;--line:#d8dee8;--group:#84aec6;--input:#9adfd7;--output:#70c7d3;
--unconnected:#ff8e86;--forward:#38b5ac;--feedback:#ff7b00;--cycle:#bf627f;--diag:#283a5c;
--hierarchy-width:clamp(230px,18vw,292px)}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;overflow:hidden;background:var(--bg);
color:var(--ink);font:13px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}
header{height:52px;display:flex;align-items:center;gap:12px;padding:7px 12px 7px 66px;
background:#14213d;color:#fff;box-shadow:0 1px 5px #0004}
h1{font-size:16px;margin:0;white-space:nowrap}.trade{color:#b8c6df}.search-wrap{margin-left:auto;
display:flex;align-items:center;gap:7px}.search-wrap label{font-size:11px;color:#cad5e8}
input{font:inherit}#search{width:min(360px,32vw);padding:6px 9px;border:1px solid #657590;border-radius:5px}
.layout{height:calc(100vh - 52px);display:grid;
grid-template-columns:52px var(--hierarchy-width) minmax(0,1fr);
gap:1px;background:var(--line)}
.layout.hierarchy-hidden{grid-template-columns:52px minmax(0,1fr)}
#toolbarLoc{grid-column:1;grid-row:1;background:var(--nav);overflow-y:auto;overflow-x:hidden;
padding:5px 4px;position:relative;z-index:15;width:52px;transition:width .16s ease,box-shadow .16s ease}
#toolbarLoc:hover,#toolbarLoc:focus-within{width:224px;box-shadow:5px 0 16px #0004}
.tool-group{border-bottom:1px solid #ffffff55;padding:3px 0;display:grid;
grid-template-columns:44px;gap:3px}
.tool-group:before{content:attr(data-label);display:none;grid-column:1/-1;color:#eafff5;
font-size:10px;font-weight:750;letter-spacing:.08em;padding:3px 4px 1px;text-transform:uppercase}
#toolbarLoc:hover .tool-group,#toolbarLoc:focus-within .tool-group{grid-template-columns:repeat(4,44px)}
#toolbarLoc:hover .tool-group:before,#toolbarLoc:focus-within .tool-group:before{display:block}
#toolbarLoc:not(:hover):not(:focus-within) .tool-group .tool:not(:first-of-type){display:none}
.tool{width:44px;height:35px;border:0;border-radius:7px;background:transparent;color:#fff;
font:600 18px/1 system-ui;cursor:pointer}.tool:hover,.tool.active{background:var(--nav2)}
.tool:disabled{opacity:.35;cursor:default}.tool.small{font-size:12px}.panel{background:#fff;overflow:auto}
.panel-title{position:sticky;top:0;z-index:5;padding:8px 10px;background:#edf2f8;
border-bottom:1px solid var(--line);font-weight:700}
#hierarchy-panel{grid-column:2;grid-row:1}.hierarchy-title{display:flex;align-items:center;
justify-content:space-between}#hierarchy-collapse{border:0;background:transparent;color:#344054;
font:700 18px/1 system-ui;cursor:pointer;padding:2px 5px;border-radius:4px}
#hierarchy-collapse:hover{background:#dce5f0}#hierarchy-reopen{position:fixed;z-index:11;left:52px;
top:78px;border:1px solid #b8c3d3;border-left:0;border-radius:0 7px 7px 0;padding:9px 7px;
background:#edf2f8;color:#26364f;font:700 11px system-ui;cursor:pointer;box-shadow:2px 2px 8px #0002;
writing-mode:vertical-rl}.layout.hierarchy-hidden #hierarchy-panel{display:none}
.layout.hierarchy-hidden #matrix-panel{grid-column:2}#tree{padding:8px 7px 20px}
.tree-group,.tree-component{margin:0;list-style:none}.tree-group{margin-left:7px}
.tree-group>summary,.component-label{padding:4px 5px;border-radius:4px;cursor:pointer;user-select:none}
.tree-group>summary:hover,.component-label:hover,.view-root{background:#e4efff}
.tree-component{margin-left:16px}.component-label.cyclic{color:#9e3659}
.variables{border-left:1px solid #d6deea;margin:2px 0 5px 11px;padding-left:10px}
.variable{padding:2px 3px;color:var(--muted);white-space:nowrap;cursor:pointer}
.variable:hover{background:#f0f5ff}.variable.input:before{content:"◀";color:#69a8df;margin-right:5px}
.variable.output:before{content:"▶";color:#58bfa2;margin-right:5px}.variable.unconnected{color:#c13932}
.collapsed-mark{float:right;color:#7d8797}.badge{font-size:9px;color:#fff;border-radius:3px;
padding:1px 3px;margin-left:3px}.dv{background:#7c3aed}.obj{background:#b42318}
.con{background:#d97706}.track{background:#16845a}
#matrix-panel{grid-column:3;grid-row:1;position:relative;overflow:auto;background:#fff}
#matrix-controls{position:fixed;z-index:12;top:60px;right:12px;width:max-content;
display:flex;align-items:center;gap:4px;padding:5px;background:#14213de8;border-radius:8px;
box-shadow:0 2px 9px #0004}
#matrix-controls button{min-width:35px;height:31px;border:1px solid #ffffff55;border-radius:5px;
background:#294368;color:#fff;font:700 15px system-ui;cursor:pointer}
#matrix-controls button:hover{background:#3e6599}.drag-handle{height:31px;display:grid;place-items:center;
padding:0 3px;color:#c9d7eb;font-size:17px;cursor:grab;user-select:none;touch-action:none}.drag-handle:active{cursor:grabbing}
.drag-handle:focus-visible{outline:2px solid #fff;border-radius:4px}#zoom-entry{display:flex;
align-items:center;gap:2px;color:#e7effc;font-size:11px}#zoom-level{width:55px;padding:5px 4px;
border:1px solid #ffffff66;border-radius:4px;background:#fff;color:#172033;text-align:right}
#zoom-mode{min-width:48px;color:#c9d7eb;font-size:10px}
#matrix{display:block;min-width:100%}.component-band{fill:var(--group);
stroke:#fff;stroke-width:2;rx:6;cursor:pointer}
.component-band.group-level{fill:#35bfe9}.component-band.solution-level{fill:#78a6c2}
.component-band.cycle{fill:#bd7890}
.component-band.aggregate{fill:#7d8797}.component-band-text{font-size:11px;fill:#10233e;
font-weight:650;pointer-events:none}.variable-band{stroke:#fff;stroke-width:1;rx:5;cursor:pointer}
.variable-band.input{fill:var(--input)}.variable-band.output{fill:var(--output)}
.variable-band.aggregate{fill:#98a5b7}.variable-band.unconnected{fill:var(--unconnected)}
.variable-band.optimization{stroke:#7c3aed;stroke-width:3}.variable-band-text{font-size:10px;
fill:#10233e;pointer-events:none}.matched{font-weight:800}.diag{fill:var(--diag);stroke:#fff;cursor:pointer}
.diag.input{fill:#69a8df}.diag.output{fill:#55bfa0}.diag.aggregate{fill:#7c8797}
.diag.unconnected{fill:var(--unconnected)}.connection{stroke:#fff;stroke-width:1;cursor:pointer}
.connection.forward{fill:var(--forward)}.connection.feedback{fill:var(--feedback)}
.connection.cycle{fill:var(--cycle)}.connection.internal{fill:#000}
.connection:hover,.connection.pinned{stroke:#111827;stroke-width:2}
.component-outline{fill:none;stroke:#4b4b4b;stroke-width:2;pointer-events:none}
.guide{fill:#1769dc12;pointer-events:none}.arrow{fill:none;stroke:#111827;stroke-width:2;
marker-end:url(#arrowhead);pointer-events:none}.arrow.pinned{stroke:#268c5e}
.arrow.internal{stroke:#ef7d72}.arrow-joint{fill:#268c5e;pointer-events:none}
.arrow-joint.internal{fill:#ef7d72}.solver-band{fill:#a9dfd4;
stroke:#fff;stroke-width:2;rx:6;cursor:pointer}.solver-text{font-size:10px;fill:#16332e;
pointer-events:none}.empty-note{font-size:16px;fill:#667085}
#info-panel{position:fixed;z-index:11;right:0;top:52px;bottom:0;width:276px;
box-shadow:-3px 0 14px #0002}.info-title{display:flex;align-items:center;justify-content:space-between}
#info-collapse{border:0;background:transparent;color:#344054;font:700 18px/1 system-ui;cursor:pointer;
padding:2px 5px;border-radius:4px}#info-collapse:hover{background:#dce5f0}#info-reopen{position:fixed;
z-index:11;right:0;top:78px;border:1px solid #b8c3d3;border-right:0;border-radius:7px 0 0 7px;
padding:9px 7px;background:#edf2f8;color:#26364f;font:700 11px system-ui;cursor:pointer;
box-shadow:-2px 2px 8px #0002;writing-mode:vertical-rl}#details{padding:10px}.detail-title{font-size:15px;
font-weight:750;margin-bottom:8px}.detail-grid{display:grid;grid-template-columns:94px 1fr;gap:5px 8px}
.key{color:var(--muted)}.connection-list{padding-left:18px}.connection-list li{margin:5px 0;
word-break:break-word}.footer-note{font-size:11px;color:var(--muted)}
.legend{display:grid;grid-template-columns:14px 1fr;gap:7px 8px;align-items:center}
.swatch{height:13px;border-radius:2px}.modal-bg{position:fixed;inset:0;background:#0007;z-index:20;
display:grid;place-items:center}.modal{background:#fff;max-width:680px;max-height:80vh;overflow:auto;
border-radius:8px;box-shadow:0 15px 50px #0008;padding:18px 22px}.modal h2{margin-top:0}
.modal button{float:right;border:0;border-radius:5px;background:#324b72;color:#fff;padding:6px 11px}
.variable-picker{width:min(720px,90vw)}.picker-controls{display:flex;gap:6px;margin:8px 0}
.picker-controls button,.picker-actions button{float:none}.picker-list{max-height:52vh;overflow:auto;
border:1px solid var(--line);border-radius:5px}.picker-heading{position:sticky;top:0;background:#edf2f8;
padding:6px 9px;font-weight:750}.picker-row{display:flex;align-items:center;gap:8px;padding:5px 9px;
border-top:1px solid #edf0f4}.picker-row:hover{background:#f4f7fb}.picker-row span{word-break:break-all}
.picker-actions{display:flex;justify-content:flex-end;gap:7px;margin-top:10px}
.hidden{display:none!important}
@media(max-width:1050px){.layout{grid-template-columns:52px var(--hierarchy-width) minmax(0,1fr)}
#info-panel,#info-reopen{display:none!important}}@media(max-width:720px){.layout{grid-template-columns:52px minmax(0,1fr)}
#hierarchy-panel,#hierarchy-reopen{display:none!important}#matrix-panel{grid-column:2}}
</style>
</head>
<body>
<header><h1 id="title"></h1><span class="trade" id="trade"></span>
  <div class="search-wrap"><label id="search-count">0 matches</label>
  <input id="search" type="search" placeholder="Search uam4x systems or variables"></div>
</header>
<div class="layout">
<nav id="toolbarLoc" aria-label="N2 controls">
  <div class="tool-group" data-label="Navigate">
    <button class="tool" id="reset-graph" title="Return home">⌂</button>
    <button class="tool" id="undo-graph" title="Back one view">◀</button>
    <button class="tool" id="redo-graph" title="Forward one view">▶</button>
  </div>
  <div class="tool-group" data-label="Hierarchy">
    <button class="tool" id="collapse-element" title="Collapse selected system">↙</button>
    <button class="tool" id="expand-element" title="Expand selected system">↗</button>
    <button class="tool small" id="collapse-all" title="Collapse all systems">COL</button>
    <button class="tool small" id="expand-all" title="Expand all systems">EXP</button>
    <button class="tool small" id="filter-target" title="Filter to selected group">TGT</button>
    <button class="tool small" id="variables-button" title="Choose visible variables for selected component">VIS</button>
  </div>
  <div class="tool-group" data-label="Connections and model">
    <button class="tool" id="hide-connections" title="Hide pinned connections">⌁</button>
    <button class="tool" id="show-all-connections" title="Show all visible connections">⑂</button>
    <button class="tool small" id="solver-button" title="Cycle solver display">SOL</button>
    <button class="tool small" id="desvars-button" title="Toggle optimization roles">f(x)</button>
    <button class="tool small" id="partials-button" title="Toggle declared component partials">JAC</button>
  </div>
  <div class="tool-group" data-label="Display">
    <button class="tool" id="font-smaller" title="Smaller text">T−</button>
    <button class="tool" id="font-larger" title="Larger text">T+</button>
    <button class="tool" id="zoom-out" title="Smaller matrix cells">−</button>
    <button class="tool" id="zoom-in" title="Larger matrix cells">+</button>
    <button class="tool small" id="model-fit" title="Fit matrix height">FIT</button>
    <button class="tool small" id="depth-button" title="Cycle displayed hierarchy depth">DEP</button>
  </div>
  <div class="tool-group" data-label="Information">
    <button class="tool" id="info-button" title="Toggle information panel">ⓘ</button>
    <button class="tool" id="legend-button" title="Show legend">◆</button>
    <button class="tool" id="question-button" title="Show interaction help">?</button>
  </div>
</nav>
<section class="panel" id="hierarchy-panel"><div class="panel-title hierarchy-title"><span>uam4x model hierarchy</span>
  <button id="hierarchy-collapse" type="button" aria-label="Collapse model hierarchy"
    title="Collapse model hierarchy">‹</button></div>
  <div id="tree"></div></section>
<button id="hierarchy-reopen" class="hidden" type="button" aria-label="Open model hierarchy">Hierarchy</button>
<section id="matrix-panel">
  <div id="matrix-controls" aria-label="Matrix zoom controls">
    <span class="drag-handle" id="matrix-controls-drag" role="button" tabindex="0"
      aria-label="Move zoom controls" title="Drag to move zoom controls">⠿</span>
    <button id="matrix-zoom-out" title="Zoom out">−</button>
    <label id="zoom-entry"><input id="zoom-level" type="number" min="3" max="941" step="1"
      value="100" aria-label="Zoom percentage"><span>%</span></label>
    <span id="zoom-mode" aria-live="polite"></span>
    <button id="matrix-fit" title="Fit matrix to the available height">Fit</button>
    <button id="matrix-zoom-in" title="Zoom in">+</button>
  </div>
  <svg id="matrix" role="img" aria-label="N-squared connection matrix"></svg>
</section>
<aside class="panel" id="info-panel"><div class="panel-title info-title"><span>N² information</span>
  <button id="info-collapse" type="button" aria-label="Collapse links panel" title="Collapse panel">›</button></div>
  <div id="details"></div></aside>
<button id="info-reopen" class="hidden" type="button" aria-label="Open links panel">Links</button>
</div>
<div id="modal-host"></div>
<script>
"use strict";
const encodedModel="__MODEL_DATA__";
const bytes=Uint8Array.from(atob(encodedModel),c=>c.charCodeAt(0));
const model=JSON.parse(new TextDecoder().decode(bytes));
const svgNS="http://www.w3.org/2000/svg";
const BOUNDARY=4,LABEL_PADDING=BOUNDARY*2,OUTER_PADDING=BOUNDARY*2;
const MIN_CELL=1,DETAIL_MIN_CELL=20,MAX_CELL=320;
const viewer=model.viewer||{};
const AUTO_COLLAPSE_ROWS=Math.max(10,Number(viewer.max_visible_rows)||120);
const DEFAULT_STROKE=1,BAND_STROKE=2,OPTIMIZATION_STROKE=3;
const svg=document.getElementById("matrix"),details=document.getElementById("details");
const byId=new Map(model.variables.map(v=>[v.id,v]));
const componentById=new Map(model.components.map(c=>[c.id,c]));
const allEdges=[...model.connections,...(model.dependencies||[])];
function activeEdges(){return state.partials?allEdges:model.connections}
const treeById=new Map(),parentById=new Map(),groupIds=[];
function indexTree(node,parent=null){
  treeById.set(node.id,node);if(parent)parentById.set(node.id,parent.id);
  if(node.kind==="group"){groupIds.push(node.id);node.children.forEach(c=>indexTree(c,node))}
}
indexTree(model.tree);
const initialRoot=treeById.has(viewer.initial_path)?viewer.initial_path:model.tree.id;
const initial={cell:34,font:13,query:"",rootId:initialRoot,collapsed:[],
  selectedNode:null,pinned:[],hiddenVariables:[],solverMode:"cycle",optimization:true,
  partials:true,info:false,hierarchy:true,depth:0};
const state={...initial,collapsed:new Set(),pinned:new Set(),hiddenVariables:new Set()};
let history=[],historyIndex=-1,lastLayout=null;
document.getElementById("title").textContent=model.title;
document.getElementById("trade").textContent=model.trade_name?`— ${model.trade_name}`:"";

function el(name,attrs={},text=null){const n=document.createElementNS(svgNS,name);
  Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==null)n.textContent=text;return n}
function htmlEl(name,cls="",text=null){const n=document.createElement(name);n.className=cls;
  if(text!==null)n.textContent=text;return n}
function snapshot(){return{cell:state.cell,font:state.font,query:state.query,rootId:state.rootId,
  collapsed:[...state.collapsed],selectedNode:state.selectedNode,pinned:[...state.pinned],
  hiddenVariables:[...state.hiddenVariables],solverMode:state.solverMode,
  optimization:state.optimization,partials:state.partials,info:state.info,
  hierarchy:state.hierarchy,depth:state.depth}}
function restore(s){Object.assign(state,s);state.collapsed=new Set(s.collapsed||[]);
  state.pinned=new Set(s.pinned||[]);state.hiddenVariables=new Set(s.hiddenVariables||[]);
  state.hierarchy=s.hierarchy!==false;
  document.getElementById("search").value=state.query||"";renderAll()}
function commit(change){change();history=history.slice(0,historyIndex+1);history.push(snapshot());
  historyIndex=history.length-1;renderAll()}
function descendantVariables(id){
  if(componentById.has(id)){const c=componentById.get(id);return [...c.inputs,...c.outputs]}
  const node=treeById.get(id),out=[];if(!node)return out;
  (function walk(n){if(n.kind==="component")out.push(...descendantVariables(n.id));
    else n.children.forEach(walk)})(node);return out
}
function badgeWrap(v){const w=htmlEl("span");
  [["DV","dv",v.design_var],["OBJ","obj",v.objective],["CON","con",v.constraint],
   ["TRACK","track",v.tracked]].forEach(([t,c,on])=>{if(on)w.appendChild(htmlEl("span",`badge ${c}`,t))});return w}
function renderTreeNode(node,parent){
  if(node.kind==="group"){
    const wrap=htmlEl("details","tree-group");wrap.open=!state.collapsed.has(node.id);
    const summary=htmlEl("summary",node.id===state.rootId?"view-root":"",node.name);
    if(node.cyclic)summary.append(`  ↻ ${node.cycle_solver||"cycle"}`);
    if(state.collapsed.has(node.id))summary.appendChild(htmlEl("span","collapsed-mark","▣"));
    summary.onclick=e=>{e.preventDefault();focusSystem(node.id)};
    summary.oncontextmenu=e=>{e.preventDefault();commit(()=>{state.selectedNode=node.id;
      state.collapsed.has(node.id)?state.collapsed.delete(node.id):state.collapsed.add(node.id)})};
    wrap.appendChild(summary);parent.appendChild(wrap);
    node.children.forEach(child=>renderTreeNode(child,wrap));return
  }
  const c=componentById.get(node.id),wrap=htmlEl("div","tree-component");
  const label=htmlEl("div",`component-label${c?.cyclic?" cyclic":""}`,node.name);
  if(state.collapsed.has(node.id))label.appendChild(htmlEl("span","collapsed-mark","▣"));
  label.onclick=()=>focusSystem(node.id);
  label.onmouseenter=()=>showComponent(c);
  label.oncontextmenu=e=>{e.preventDefault();state.selectedNode=node.id;
    if(e.altKey)variableVisibilityDialog(node.id);
    else commit(()=>state.collapsed.has(node.id)?state.collapsed.delete(node.id):state.collapsed.add(node.id))};
  wrap.appendChild(label);
  if(c&&!state.collapsed.has(node.id)){const vars=htmlEl("div","variables");
    [...c.inputs,...c.outputs].forEach(id=>{const v=byId.get(id);
      if(!v||state.hiddenVariables.has(id))return;
      const disconnected=v.direction==="input"&&!v.connected?" unconnected":"";
      const row=htmlEl("div",`variable ${v.direction}${disconnected}`,v.name);row.appendChild(badgeWrap(v));
      row.onclick=()=>{state.selectedNode=v.id;showVariable(v)};vars.appendChild(row)});wrap.appendChild(vars)}
  parent.appendChild(wrap)
}
function renderTree(){const host=document.getElementById("tree");host.replaceChildren();renderTreeNode(model.tree,host)}
function aggregateNode(id,kind){const c=componentById.get(id),t=treeById.get(id);
  return{id,name:c?.name||t?.name||id,kind:"aggregate",aggregateKind:kind,component:id,
    direction:"aggregate",connected:true,cyclic:Boolean(c?.cyclic||t?.cyclic)}}
function systemPath(id){
  const path=[];let current=id;
  while(current){path.push(current);current=parentById.get(current)}
  path.reverse();
  if(path[0]!==model.tree.id)path.unshift(model.tree.id);
  const focusedIndex=path.indexOf(state.rootId);
  return focusedIndex>=0?path.slice(focusedIndex):path
}
const labelCanvas=document.createElement("canvas"),labelContext=labelCanvas.getContext("2d");
function measuredTextWidth(text,fontSize=state.font,fontWeight=600){
  labelContext.font=`${fontWeight} ${fontSize}px system-ui`;
  return Math.ceil(labelContext.measureText(String(text)).width)
}
function systemLabel(id){
  return componentById.get(id)?.name||treeById.get(id)?.name||id
}
function diagramFonts(cell){
  const variableFont=Math.min(16,state.font+3,
    Math.max(state.font,Math.floor(cell*.26)));
  return{variableFont,systemFont:Math.max(11,variableFont-1),
    solverFont:Math.max(10,variableFont-2)}
}
function visibleComponentRows(component){
  return [...component.inputs,...component.outputs].filter(id=>!state.hiddenVariables.has(id)).length
}
function displayedRowCount(node,collapsed){
  if(node.kind==="component"){
    if(collapsed.has(node.id)&&node.id!==state.rootId)return 1;
    const c=componentById.get(node.id);return Math.max(1,c?visibleComponentRows(c):0)
  }
  if(collapsed.has(node.id)&&node.id!==state.rootId)return 1;
  return node.children.reduce((sum,child)=>sum+displayedRowCount(child,collapsed),0)
}
function autoCollapsedFor(rootId){
  const root=treeById.get(rootId);const collapsed=new Set();
  if(!viewer.auto_collapse||!root||root.kind==="component")return collapsed;
  const candidates=[];
  (function walk(node,depth){
    if(node.id!==rootId)candidates.push({node,depth});
    if(node.kind==="group")node.children.forEach(child=>walk(child,depth+1))
  })(root,0);
  function reduceWith(items){
    while(displayedRowCount(root,collapsed)>AUTO_COLLAPSE_ROWS){
      let best=null,bestSaving=0;
      items.forEach(item=>{
        if(collapsed.has(item.node.id))return;
        let parent=parentById.get(item.node.id);
        while(parent&&parent!==rootId){if(collapsed.has(parent))return;parent=parentById.get(parent)}
        const saving=displayedRowCount(item.node,collapsed)-1;
        if(saving>bestSaving||(saving===bestSaving&&best&&item.depth>best.depth)){
          best=item;bestSaving=saving
        }
      });
      if(!best||bestSaving<=0)break;
      collapsed.add(best.node.id)
    }
  }
  // Preserve the group hierarchy first, as OpenMDAO does: aggregate large
  // components before replacing an entire subgroup with one matrix row.
  reduceWith(candidates.filter(item=>item.node.kind==="component"));
  reduceWith(candidates.filter(item=>item.node.kind==="group"));
  return collapsed
}
function applyAutomaticCollapse(rootId=state.rootId){
  state.collapsed=autoCollapsedFor(rootId)
}
function boundaryFor(cell){return Math.min(BOUNDARY,Math.max(.15,cell*.18))}
function strokeFor(cell,normal){return cell<6?0:normal}
function layoutMetrics(nodes,paths,systemDepth,cell=state.cell){
  const fonts=diagramFonts(cell);
  const compact=cell<state.font+6;
  const systemWidths=Array(systemDepth).fill(compact?32:56);
  if(!compact)paths.forEach(path=>path.forEach((id,depth)=>{
    systemWidths[depth]=Math.max(systemWidths[depth],
      measuredTextWidth(systemLabel(id),fonts.systemFont,650)+LABEL_PADDING*2)
  }));
  const variableWidth=compact?48:Math.max(96,...nodes.map(node=>
    measuredTextWidth(node.name,fonts.variableFont,600)+LABEL_PADDING*2));
  let solverWidth=compact?48:80;
  if(state.solverMode!=="none"){
    nodes.forEach(node=>{
      const c=componentById.get(node.component),group=treeById.get(node.component);
      const solver=group?.cycle_solver||c?.cycle_solver;
      const text=state.solverMode==="cycle"?(solver||"Sequential"):`Order ${c?.order??"group"}`;
      if(!compact)solverWidth=Math.max(solverWidth,
        measuredTextWidth(text,fonts.solverFont,500)+LABEL_PADDING*2)
    })
  }
  return{systemWidths,variableWidth,solverWidth,compact,...fonts}
}
function geometry(cell,nodeCount,metrics,solverMode=state.solverMode){
  const systemWidths=metrics.systemWidths;
  const variableWidth=metrics.variableWidth;
  const solverWidth=solverMode==="none"?0:metrics.solverWidth;
  const hierarchyWidth=systemWidths.reduce((sum,width)=>sum+width,0);
  const matrixWidth=nodeCount*cell;
  const totalWidth=OUTER_PADDING+hierarchyWidth+variableWidth+BOUNDARY+matrixWidth+
    (solverWidth?BOUNDARY+solverWidth:0)+OUTER_PADDING;
  const totalHeight=OUTER_PADDING+nodeCount*cell+OUTER_PADDING;
  return{systemWidths,variableWidth,solverWidth,totalWidth,totalHeight}
}
function buildLayout(){
  const nodes=[],displayForVar=new Map(),root=treeById.get(state.rootId)||model.tree;
  function addAggregate(id,kind){nodes.push(aggregateNode(id,kind));
    descendantVariables(id).forEach(v=>displayForVar.set(v,id))}
  function walk(node,depth){
    if(node.kind==="group"){
      if((state.collapsed.has(node.id)&&node.id!==state.rootId)||(state.depth&&depth>=state.depth)){
        addAggregate(node.id,"group");return}
      node.children.forEach(c=>walk(c,depth+1));return
    }
    const c=componentById.get(node.id);if(!c)return;
    if(state.collapsed.has(node.id)){addAggregate(node.id,"component");return}
    [...c.inputs,...c.outputs].forEach(id=>{const v=byId.get(id);
      if(v&&!state.hiddenVariables.has(id)){nodes.push({...v,kind:"variable"});displayForVar.set(id,id)}})
  }
  walk(root,0);let visible=nodes;const q=state.query.trim().toLowerCase();
  if(q){const matches=new Set(nodes.filter(n=>`${n.id} ${n.name} ${n.dtype||""}`.toLowerCase().includes(q)).map(n=>n.id));
    activeEdges().forEach(c=>{const a=displayForVar.get(c.source),b=displayForVar.get(c.target);
      if(matches.has(a)&&b)matches.add(b);if(matches.has(b)&&a)matches.add(a)});
    visible=nodes.filter(n=>matches.has(n.id))}
  const index=new Map(visible.map((n,i)=>[n.id,i])),groups=new Map();
  activeEdges().forEach(link=>{const source=displayForVar.get(link.source),target=displayForVar.get(link.target);
    if(!index.has(source)||!index.has(target)||source===target)return;
    const key=`${source}|${target}`;if(!groups.has(key))groups.set(key,{key,source,target,links:[]});
    groups.get(key).links.push(link)});
  return{nodes:visible,index,groups}
}
function renderMatrix(){
  svg.replaceChildren();const defs=el("defs"),marker=el("marker",{id:"arrowhead",markerWidth:8,
    markerHeight:8,refX:7,refY:3,orient:"auto"});marker.appendChild(el("path",
    {d:"M0,0 L0,6 L8,3 z",fill:"#111827"}));defs.appendChild(marker);svg.appendChild(defs);
  const fullLayout=buildLayout();let nodes=fullLayout.nodes;
  document.getElementById("search-count").textContent=`${nodes.length} matches`;
  const index=new Map(nodes.map((n,i)=>[n.id,i]));
  const groups=new Map([...fullLayout.groups].filter(([,group])=>
    index.has(group.source)&&index.has(group.target)));
  const layout={...fullLayout,nodes,index,groups};lastLayout=layout;
  const paths=nodes.map(n=>systemPath(n.kind==="aggregate"?n.id:n.component));
  const maxSystemDepth=Math.max(1,...paths.map(path=>path.length));
  const metrics=layoutMetrics(nodes,paths,maxSystemDepth);
  const dims=geometry(state.cell,nodes.length,metrics);
  const systemWidths=dims.systemWidths,vWidth=dims.variableWidth,gap=BOUNDARY;
  const cLeft=OUTER_PADDING;
  const systemOffsets=[];let nextSystemLeft=cLeft;
  systemWidths.forEach(width=>{systemOffsets.push(nextSystemLeft);nextSystemLeft+=width});
  const hierarchyRight=nextSystemLeft;
  const variableRight=hierarchyRight+vWidth;
  const left=variableRight+gap;
  const top=OUTER_PADDING,size=state.cell,matrixWidth=nodes.length*size;
  const detailBoundary=boundaryFor(size);
  const solverWidth=dims.solverWidth,solverLeft=left+matrixWidth+BOUNDARY;
  svg.setAttribute("width",Math.max(document.getElementById("matrix-panel").clientWidth,dims.totalWidth));
  svg.setAttribute("height",Math.max(80,dims.totalHeight));
  if(!nodes.length){svg.appendChild(el("text",{x:30,y:50,class:"empty-note"},"No nodes match this view."));return}
  for(let depth=0;depth<maxSystemDepth;depth++){
    let start=0;
    while(start<nodes.length){
      const systemId=paths[start][depth];
      if(!systemId){start++;continue}
      let end=start+1;
      while(end<nodes.length&&paths[end][depth]===systemId)end++;
      const c=componentById.get(systemId),node=treeById.get(systemId);
      const y=top+start*size,h=(end-start)*size;
      const levelClass=systemId===model.tree.id?" solution-level":node?.kind==="group"?" group-level":"";
      const aggregate=nodes[start].kind==="aggregate"&&nodes[start].id===systemId;
      const systemLeft=systemOffsets[depth],systemWidth=systemWidths[depth];
      const bandStroke=strokeFor(size,BAND_STROKE);
      const bandInset=(detailBoundary+bandStroke)/2;
      const band=el("rect",{x:systemLeft+bandInset,y:y+bandInset,
        width:systemWidth-bandInset*2,height:h-bandInset*2,
        style:`stroke-width:${bandStroke}`,
        class:`component-band${levelClass}${c?.cycle_id!=null||node?.cyclic?" cycle":""}${aggregate?" aggregate":""}`});
      band.appendChild(el("title",{},systemLabel(systemId)));
      band.onclick=()=>focusSystem(systemId);
      band.onmouseenter=()=>c?showComponent(c):showGroup(node);
      band.oncontextmenu=e=>{e.preventDefault();
        if(e.altKey&&componentById.has(systemId)){state.selectedNode=systemId;variableVisibilityDialog(systemId)}
        else if(systemId!==model.tree.id)toggleCollapse(systemId)};
      svg.appendChild(band);
      const label=systemLabel(systemId);
      if(!metrics.compact&&h>=metrics.systemFont+detailBoundary*2)svg.appendChild(el("text",{
        x:systemLeft+systemWidth/2,y:y+h/2+metrics.systemFont*.35,
        "text-anchor":"middle","font-size":metrics.systemFont,
        class:"component-band-text"},label));
      start=end
    }
  }
  let start=0;
  while(start<nodes.length){const comp=nodes[start].component;let end=start+1;
    while(end<nodes.length&&nodes[end].component===comp)end++;
    const c=componentById.get(comp),node=treeById.get(comp),y=top+start*size,h=(end-start)*size;
    if(solverWidth){const solver=node?.cycle_solver||c?.cycle_solver;
      const text=state.solverMode==="cycle"?(solver||"Sequential"):`Order ${c?.order??"group"}`;
      const bandStroke=strokeFor(size,BAND_STROKE);
      const bandInset=(detailBoundary+bandStroke)/2;
      const sb=el("rect",{x:solverLeft+bandInset,y:y+bandInset,
        width:solverWidth-bandInset*2,height:h-bandInset*2,
        style:`stroke-width:${bandStroke}`,class:"solver-band"});
      sb.appendChild(el("title",{},text));
      sb.onclick=()=>showMessage(`${c?.id||node?.id}: ${text}`);svg.appendChild(sb);
      if(!metrics.compact&&h>=metrics.solverFont+detailBoundary*2)svg.appendChild(el("text",{x:solverLeft+solverWidth/2,
        y:y+h/2+metrics.solverFont*.35,"text-anchor":"middle",
        "font-size":metrics.solverFont,class:"solver-text"},text))}
    start=end
  }
  nodes.forEach((n,i)=>{const y=top+i*size,x=left+i*size;
    const vLeft=cLeft+systemWidths.slice(0,paths[i].length).reduce((sum,width)=>sum+width,0);
    let bandClass=`variable-band ${n.direction}`;if(n.direction==="input"&&!n.connected)bandClass+=" unconnected";
    if(state.optimization&&(n.design_var||n.constraint||n.objective))bandClass+=" optimization";
    const optimization=bandClass.includes(" optimization");
    const bandStroke=strokeFor(size,optimization?OPTIMIZATION_STROKE:DEFAULT_STROKE);
    const bandInset=(detailBoundary+bandStroke)/2;
    const band=el("rect",{x:vLeft+bandInset,y:y+bandInset,
      width:variableRight-vLeft-bandInset*2,height:size-bandInset*2,
      style:`stroke-width:${bandStroke}`,class:bandClass});
    band.appendChild(el("title",{},n.id));
    band.onmouseenter=()=>showNode(n);band.onclick=()=>{
      if(n.kind==="aggregate")focusSystem(n.id);
      else{state.selectedNode=n.id;showNode(n)}
    };
    svg.appendChild(band);
    if(!metrics.compact)svg.appendChild(el("text",{x:vLeft+(variableRight-vLeft)/2,
      y:y+size/2+metrics.variableFont*.35,"text-anchor":"middle",
      "font-size":metrics.variableFont,
      class:`variable-band-text${state.query&&n.id.toLowerCase().includes(state.query.toLowerCase())?" matched":""}`},n.name));
    let cls=`diag ${n.direction}`;if(n.direction==="input"&&!n.connected)cls+=" unconnected";
    const cellStroke=strokeFor(size,DEFAULT_STROKE);
    const cellInset=(detailBoundary+cellStroke)/2;
    const d=el("rect",{x:x+cellInset,y:y+cellInset,
      width:size-cellInset*2,height:size-cellInset*2,
      style:`stroke-width:${cellStroke}`,class:cls});
    d.appendChild(el("title",{},n.id));
    d.onmouseenter=()=>{showNode(n);showNodeLinks(n.id,left,top,size)};
    d.onmouseleave=()=>removeHoverOverlays();d.onclick=()=>toggleNodeConnections(n.id);
    svg.appendChild(d)
  });
  layout.groups.forEach(g=>{if(!index.has(g.source)||!index.has(g.target))return;
    const si=index.get(g.source),ti=index.get(g.target),cycle=g.links.some(l=>l.cycle);
    const internal=g.links.every(l=>l.internal);
    const cellStroke=strokeFor(size,DEFAULT_STROKE);
    const cellInset=(detailBoundary+cellStroke)/2;
    const cell=el("rect",{x:left+ti*size+cellInset,y:top+si*size+cellInset,
      width:size-cellInset*2,height:size-cellInset*2,
      style:`stroke-width:${cellStroke}`,
      class:`connection ${internal?"internal":cycle?"cycle":si<ti?"forward":"feedback"}${state.pinned.has(g.key)?" pinned":""}`});
    cell.onmouseenter=()=>{showConnections(g);drawArrow(si,ti,left,top,size,false,internal)};
    cell.onmouseleave=()=>removeHoverOverlays();cell.onclick=()=>commit(()=>state.pinned.has(g.key)?state.pinned.delete(g.key):state.pinned.add(g.key));
    svg.appendChild(cell);if(g.links.length>1&&size>=12)svg.appendChild(el("text",{x:left+ti*size+size/2,
      y:top+si*size+size/2+Math.min(state.font,size*.35),"text-anchor":"middle",
      "font-size":Math.max(8,size*.45),fill:"#fff",
      "pointer-events":"none"},String(g.links.length)))
  });
  let outlineStart=0;
  while(outlineStart<nodes.length){
    const componentId=nodes[outlineStart].component;
    let outlineEnd=outlineStart+1;
    while(outlineEnd<nodes.length&&nodes[outlineEnd].component===componentId)outlineEnd++;
    if(componentById.has(componentId)){
      const span=(outlineEnd-outlineStart)*size;
      svg.appendChild(el("rect",{x:left+outlineStart*size,y:top+outlineStart*size,
        width:span,height:span,style:`stroke-width:${Math.max(.35,Math.min(2,size*.12))}`,
        class:"component-outline"}))
    }
    outlineStart=outlineEnd
  }
  layout.groups.forEach(g=>{if(state.pinned.has(g.key)&&index.has(g.source)&&index.has(g.target))
    drawArrow(index.get(g.source),index.get(g.target),left,top,size,true,g.links.every(l=>l.internal))})
}
function drawArrow(source,target,left,top,size,pinned,internal=false){
  const sx=left+source*size+size/2,sy=top+source*size+size/2;
  const tx=left+target*size+size/2,ty=top+target*size+size/2;
  svg.appendChild(el("rect",{x:left,y:top+source*size,width:(target+1)*size,height:size,
    class:`guide ${pinned?"pinned-overlay":"hover-overlay"}`}));
  svg.appendChild(el("rect",{x:left+target*size,y:top,width:size,height:(source+1)*size,
    class:`guide ${pinned?"pinned-overlay":"hover-overlay"}`}));
  svg.appendChild(el("path",{d:`M${sx},${sy} L${tx},${sy} L${tx},${ty}`,
    class:`arrow${internal?" internal":""}${pinned?" pinned pinned-overlay":" hover-overlay"}`}));
  svg.appendChild(el("circle",{cx:tx,cy:sy,r:Math.max(3,size*.035),
    class:`arrow-joint${internal?" internal":""} ${pinned?"pinned-overlay":"hover-overlay"}`}))
}
function removeHoverOverlays(){svg.querySelectorAll(".hover-overlay").forEach(n=>n.remove())}
function showNodeLinks(id,left,top,size){if(!lastLayout)return;lastLayout.groups.forEach(g=>{
  if(g.source===id||g.target===id)drawArrow(lastLayout.index.get(g.source),lastLayout.index.get(g.target),
    left,top,size,false,g.links.every(l=>l.internal))})}
function detailGrid(items){const g=htmlEl("div","detail-grid");items.forEach(([k,v])=>{
  g.appendChild(htmlEl("div","key",k));g.appendChild(htmlEl("div","",String(v??"—")))});return g}
function showNode(n){if(n.kind==="aggregate")showGroup(treeById.get(n.id)||componentById.get(n.id));
  else showVariable(byId.get(n.id))}
function showGroup(g){if(!g)return;details.replaceChildren(htmlEl("div","detail-title",g.id));
  details.appendChild(detailGrid([["Kind",g.kind||"group"],["Variables",descendantVariables(g.id).length],
    ["Cyclic",Boolean(g.cyclic)],["Cycle solver",g.cycle_solver],["Interaction","Double/right-click to collapse"]]))}
function showComponent(c){if(!c)return;details.replaceChildren(htmlEl("div","detail-title",c.id));
  details.appendChild(detailGrid([["Plugin",c.plugin_class],["Inputs",c.inputs.length],["Outputs",c.outputs.length],
    ["Visible variables",visibleComponentRows(c)],
    ["Execution order",c.order],["Cycle",c.cycle_id==null?"No":`Cycle ${c.cycle_id+1}`],["Solver",c.cycle_solver]]))}
function showVariable(v){if(!v)return;details.replaceChildren(htmlEl("div","detail-title",v.id));
  const items=[["Direction",v.direction],["Type",v.dtype],["Shape",v.shape?v.shape.join(" × "):"scalar"],
    ["Connected",v.connected],["Tracked",v.tracked],["Design variable",v.design_var],
    ["Constraint",v.constraint],["Objective",v.objective]];
  if(Object.hasOwn(v,"value"))items.splice(3,0,["Declared value",JSON.stringify(v.value).slice(0,180)]);
  details.appendChild(detailGrid(items))}
function showConnections(g){details.replaceChildren(htmlEl("div","detail-title",`${g.source} → ${g.target}`));
  details.appendChild(detailGrid([["Links",g.links.length],["Classification",g.links.every(l=>l.internal)?
    "Internal input → output dependency":g.links.some(l=>l.cycle)?"Cycle":
    g.links.every(l=>l.forward)?"Feed-forward":"Feedback"]]));const list=htmlEl("ol","connection-list");
  g.links.forEach(l=>list.appendChild(htmlEl("li","",`${l.source}${l.source_indices||""} → ${l.target}`)));
  details.appendChild(list)}
function showDefault(){details.innerHTML=`<div class="detail-title">uam4x N²</div>
  <p>Hover over cells to trace links. Click a connection to pin it. Click a group to zoom;
  right-click a group or component to collapse it. Alt-right-click a component to choose its visible variables.</p>
  <p class="footer-note">Generated from the native
  uam4x solution without executing the model.</p>`}
function showMessage(m){details.replaceChildren(htmlEl("div","detail-title","Viewer notice"),htmlEl("p","",m))}
function toggleCollapse(id){commit(()=>state.collapsed.has(id)?state.collapsed.delete(id):state.collapsed.add(id))}
function toggleNodeConnections(id){
  if(!lastLayout)return;
  const attached=[...lastLayout.groups.values()].filter(group=>group.source===id||group.target===id);
  if(!attached.length){
    const node=lastLayout.nodes.find(item=>item.id===id);
    if(node){state.selectedNode=id;showNode(node)}
    return
  }
  const allPinned=attached.every(group=>state.pinned.has(group.key));
  commit(()=>attached.forEach(group=>allPinned?state.pinned.delete(group.key):state.pinned.add(group.key)))
}
function focusSystem(id){
  if(!treeById.has(id))return;
  if(state.rootId===id){const c=componentById.get(id);c?showComponent(c):showGroup(treeById.get(id));return}
  commit(()=>{state.rootId=id;state.selectedNode=id;state.pinned.clear();
    applyAutomaticCollapse(id);applyPreferredScale()});
  const panel=document.getElementById("matrix-panel");panel.scrollTop=0;panel.scrollLeft=0
}
function renderAll(){const layout=document.querySelector(".layout");
  layout.classList.toggle("hierarchy-hidden",!state.hierarchy);
  document.getElementById("hierarchy-reopen").classList.toggle("hidden",state.hierarchy);
  renderTree();renderMatrix();document.getElementById("info-panel").classList.toggle("hidden",!state.info);
  document.getElementById("hierarchy-collapse").setAttribute("aria-expanded",String(state.hierarchy));
  if(matrixControls.style.left){const controlsRect=matrixControls.getBoundingClientRect();
    moveMatrixControls(controlsRect.left,controlsRect.top)}
  document.getElementById("info-reopen").classList.toggle("hidden",state.info);
  document.getElementById("undo-graph").disabled=historyIndex<=0;
  document.getElementById("redo-graph").disabled=historyIndex>=history.length-1;
  document.getElementById("solver-button").textContent=state.solverMode==="cycle"?"CYC":state.solverMode==="execution"?"ORD":"SOL";
  document.getElementById("desvars-button").classList.toggle("active",state.optimization);
  document.getElementById("partials-button").classList.toggle("active",state.partials);
  document.getElementById("info-button").classList.toggle("active",state.info);
  const focusedComponent=componentById.has(state.rootId);
  const compact=state.cell<state.font+6;
  const mode=compact?" · overview":focusedComponent?" · detail":"";
  const zoomLevel=document.getElementById("zoom-level");
  if(document.activeElement!==zoomLevel)zoomLevel.value=String(Math.round(state.cell/34*100));
  document.getElementById("zoom-mode").textContent=mode.replace(" · ","");
  document.getElementById("variables-button").disabled=!selectedComponent()}
function selectedSystem(){const id=state.selectedNode;return componentById.has(id)||treeById.get(id)?.kind==="group"?id:null}
function selectedComponent(){
  if(componentById.has(state.selectedNode))return state.selectedNode;
  return componentById.has(state.rootId)?state.rootId:null
}
document.getElementById("search").oninput=e=>{state.query=e.target.value;renderMatrix()};
document.getElementById("reset-graph").onclick=()=>commit(()=>{Object.assign(state,{...initial,
  collapsed:new Set(),pinned:new Set(),hiddenVariables:new Set()});applyAutomaticCollapse(initial.rootId);
  document.getElementById("search").value="";applyPreferredScale()});
document.getElementById("undo-graph").onclick=()=>{if(historyIndex>0)restore(history[--historyIndex])};
document.getElementById("redo-graph").onclick=()=>{if(historyIndex<history.length-1)restore(history[++historyIndex])};
document.getElementById("collapse-element").onclick=()=>{const id=selectedSystem();if(id)commit(()=>state.collapsed.add(id))};
document.getElementById("expand-element").onclick=()=>{const id=selectedSystem();if(id)commit(()=>state.collapsed.delete(id))};
document.getElementById("collapse-all").onclick=()=>commit(()=>{componentById.forEach((_,id)=>state.collapsed.add(id));
  groupIds.filter(id=>id!==state.rootId).forEach(id=>state.collapsed.add(id))});
document.getElementById("expand-all").onclick=()=>commit(()=>state.collapsed.clear());
document.getElementById("filter-target").onclick=()=>{const id=selectedSystem();if(id)focusSystem(id)};
document.getElementById("variables-button").onclick=()=>{const id=selectedComponent();if(id)variableVisibilityDialog(id)};
document.getElementById("hide-connections").onclick=()=>commit(()=>state.pinned.clear());
document.getElementById("show-all-connections").onclick=()=>commit(()=>{if(lastLayout)lastLayout.groups.forEach((_,k)=>state.pinned.add(k))});
document.getElementById("solver-button").onclick=()=>commit(()=>state.solverMode=
  state.solverMode==="cycle"?"execution":state.solverMode==="execution"?"none":"cycle");
document.getElementById("desvars-button").onclick=()=>commit(()=>state.optimization=!state.optimization);
document.getElementById("partials-button").onclick=()=>commit(()=>{state.partials=!state.partials;state.pinned.clear()});
document.getElementById("font-smaller").onclick=()=>commit(()=>{
  state.font=Math.max(10,state.font-1);applyPreferredScale()
});
document.getElementById("font-larger").onclick=()=>commit(()=>{state.font=Math.min(16,state.font+1);
  applyPreferredScale()});
function zoomBy(factor){commit(()=>{
  state.cell=Math.max(MIN_CELL,Math.min(MAX_CELL,state.cell*factor))
})}
document.getElementById("zoom-out").onclick=()=>zoomBy(1/1.15);
document.getElementById("zoom-in").onclick=()=>zoomBy(1.15);
function setZoomPercent(raw){
  const percent=Number.parseFloat(raw);
  if(!Number.isFinite(percent)){renderAll();return}
  commit(()=>{state.cell=Math.max(MIN_CELL,Math.min(MAX_CELL,34*percent/100))})
}
const zoomInput=document.getElementById("zoom-level");
zoomInput.onchange=()=>setZoomPercent(zoomInput.value);
zoomInput.onkeydown=e=>{if(e.key==="Enter"){e.preventDefault();zoomInput.blur()}
  else if(e.key==="Escape"){zoomInput.value=String(Math.round(state.cell/34*100));zoomInput.blur()}};
const matrixControls=document.getElementById("matrix-controls"),dragHandle=document.getElementById("matrix-controls-drag");
function moveMatrixControls(left,top){
  const bounds=document.getElementById("matrix-panel").getBoundingClientRect();
  const maxLeft=Math.max(bounds.left,bounds.right-matrixControls.offsetWidth);
  const maxTop=Math.max(bounds.top,bounds.bottom-matrixControls.offsetHeight);
  matrixControls.style.left=`${Math.max(bounds.left,Math.min(maxLeft,left))}px`;
  matrixControls.style.top=`${Math.max(bounds.top,Math.min(maxTop,top))}px`;
  matrixControls.style.right="auto"
}
dragHandle.onpointerdown=e=>{
  if(e.button!==0)return;e.preventDefault();dragHandle.setPointerCapture(e.pointerId);
  const start=matrixControls.getBoundingClientRect(),originX=e.clientX,originY=e.clientY;
  const move=event=>moveMatrixControls(start.left+event.clientX-originX,start.top+event.clientY-originY);
  const finish=()=>{dragHandle.onpointermove=null;dragHandle.onpointerup=null;dragHandle.onpointercancel=null};
  dragHandle.onpointermove=move;dragHandle.onpointerup=finish;dragHandle.onpointercancel=finish
};
dragHandle.onkeydown=e=>{if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key))return;
  e.preventDefault();const rect=matrixControls.getBoundingClientRect(),step=e.shiftKey?20:5;
  moveMatrixControls(rect.left+(e.key==="ArrowLeft"?-step:e.key==="ArrowRight"?step:0),
    rect.top+(e.key==="ArrowUp"?-step:e.key==="ArrowDown"?step:0))};
window.addEventListener("resize",()=>{if(matrixControls.style.left){const rect=matrixControls.getBoundingClientRect();
  moveMatrixControls(rect.left,rect.top)}});
function fittedScale(){
  const nodes=buildLayout().nodes,n=Math.max(1,nodes.length);
  const paths=nodes.map(node=>systemPath(node.kind==="aggregate"?node.id:node.component));
  const depth=Math.max(1,...paths.map(path=>path.length));
  const panel=document.getElementById("matrix-panel");
  const availableWidth=Math.max(240,panel.clientWidth-8),availableHeight=Math.max(160,panel.clientHeight-8);
  const minimum=MIN_CELL;
  let low=minimum,high=MAX_CELL,fitted=minimum;
  while(low<=high){
    const candidate=Math.floor((low+high)/2);
    const metrics=layoutMetrics(nodes,paths,depth,candidate);
    const dims=geometry(candidate,n,metrics);
    if(dims.totalWidth<=availableWidth&&dims.totalHeight<=availableHeight){
      fitted=candidate;low=candidate+1
    }else high=candidate-1
  }
  return fitted
}
function applyFittedScale(){state.cell=fittedScale()}
function applyPreferredScale(){
  const fitted=fittedScale();
  state.cell=componentById.has(state.rootId)?Math.max(DETAIL_MIN_CELL,fitted):fitted
}
function fitMatrix(){commit(()=>{state.info=false;applyFittedScale()});
  const panel=document.getElementById("matrix-panel");panel.scrollTop=0;panel.scrollLeft=0;
}
document.getElementById("model-fit").onclick=fitMatrix;
document.getElementById("matrix-zoom-out").onclick=()=>zoomBy(1/1.15);
document.getElementById("matrix-zoom-in").onclick=()=>zoomBy(1.15);
document.getElementById("matrix-fit").onclick=fitMatrix;
document.getElementById("depth-button").onclick=()=>commit(()=>state.depth=(state.depth+1)%(Math.max(2,groupIds.length+1)));
document.getElementById("info-button").onclick=()=>commit(()=>state.info=!state.info);
document.getElementById("info-collapse").onclick=()=>commit(()=>state.info=false);
document.getElementById("info-reopen").onclick=()=>commit(()=>state.info=true);
document.getElementById("hierarchy-collapse").onclick=()=>commit(()=>state.hierarchy=false);
document.getElementById("hierarchy-reopen").onclick=()=>commit(()=>state.hierarchy=true);
function modal(title,body){const host=document.getElementById("modal-host"),bg=htmlEl("div","modal-bg"),box=htmlEl("div","modal");
  box.innerHTML=`<h2>${title}</h2>${body}`;const close=htmlEl("button","","Close");close.onclick=()=>host.replaceChildren();
  bg.onclick=e=>{if(e.target===bg)host.replaceChildren()};box.appendChild(close);bg.appendChild(box);host.replaceChildren(bg)}
function variableVisibilityDialog(componentId){
  const component=componentById.get(componentId);if(!component)return;
  const host=document.getElementById("modal-host"),bg=htmlEl("div","modal-bg");
  const box=htmlEl("div","modal variable-picker"),heading=htmlEl("h2","",`${component.name} variables`);
  box.appendChild(heading);box.appendChild(htmlEl("p","",
    "Clear variables to suppress them from the hierarchy and matrix. This changes only the diagram."));
  const controls=htmlEl("div","picker-controls"),list=htmlEl("div","picker-list");
  const checkboxes=[];
  function addButton(label,visible){const button=htmlEl("button","",label);
    button.onclick=()=>checkboxes.forEach(item=>item.checkbox.checked=visible);
    controls.appendChild(button)}
  addButton("Select all",true);addButton("Select none",false);
  [["Inputs only","input"],["Outputs only","output"]].forEach(([label,direction])=>{
    const button=htmlEl("button","",label);button.onclick=()=>checkboxes.forEach(item=>
      item.checkbox.checked=item.variable.direction===direction);controls.appendChild(button)});
  box.appendChild(controls);
  [["Inputs",component.inputs],["Outputs",component.outputs]].forEach(([title,ids])=>{
    list.appendChild(htmlEl("div","picker-heading",`${title} (${ids.length})`));
    ids.forEach(id=>{const variable=byId.get(id);if(!variable)return;
      const row=htmlEl("label","picker-row"),checkbox=document.createElement("input");
      checkbox.type="checkbox";checkbox.checked=!state.hiddenVariables.has(id);
      row.appendChild(checkbox);row.appendChild(htmlEl("span","",variable.name));
      checkboxes.push({checkbox,variable});list.appendChild(row)})
  });
  box.appendChild(list);
  const actions=htmlEl("div","picker-actions"),cancel=htmlEl("button","","Cancel"),apply=htmlEl("button","","Apply");
  cancel.onclick=()=>host.replaceChildren();apply.onclick=()=>{const hidden=new Set(state.hiddenVariables);
    checkboxes.forEach(({checkbox,variable})=>checkbox.checked?hidden.delete(variable.id):hidden.add(variable.id));
    host.replaceChildren();commit(()=>{state.hiddenVariables=hidden;state.pinned.clear();applyPreferredScale()})};
  actions.append(cancel,apply);box.appendChild(actions);bg.appendChild(box);
  bg.onclick=e=>{if(e.target===bg)host.replaceChildren()};host.replaceChildren(bg)
}
document.getElementById("legend-button").onclick=()=>modal("uam4x N² legend",`<div class="legend">
  <span class="swatch" style="background:var(--input)"></span><span>Input</span>
  <span class="swatch" style="background:var(--output)"></span><span>Output</span>
  <span class="swatch" style="background:var(--forward)"></span><span>Feed-forward link</span>
  <span class="swatch" style="background:var(--feedback)"></span><span>Feedback link</span>
  <span class="swatch" style="background:var(--cycle)"></span><span>Cycle link</span>
  <span class="swatch" style="background:var(--unconnected)"></span><span>Unconnected input</span></div>`);
document.getElementById("question-button").onclick=()=>modal("Using the uam4x N² viewer",`<p><b>Navigate:</b>
  click a group or component band to make it the diagram root and refit the focused view.
  Use Home, Back, and Forward to navigate view history.</p>
  <p><b>Collapse:</b> right-click a group or component, double-click a component band, or use the collapse controls.
  Aggregated matrix cells preserve all links crossing the collapsed boundary.</p>
  <p><b>Variable visibility:</b> select a component and press VIS, or Alt-right-click its hierarchy or
  matrix band. Suppression affects only this diagram and is tracked by Back and Forward.</p>
  <p><b>Connections:</b> hover a diagonal or off-diagonal cell to trace links. Click a link cell to pin it.
  Clicking a diagonal variable square pins or releases all arrows attached to that variable.
  Hide and Show All control persistent arrows. JAC independently shows or hides declared component partials.</p>
  <p><b>Large models:</b> oversized groups initially aggregate their largest components or subgroups.
  A focused component is never aggregated or truncated; it opens at a readable detail scale and scrolls. Manual zoom-out may
  enter overview mode, while Fit deliberately scales the complete focused component into the viewport.</p>
  <p><b>uam4x mapping:</b> systems are Solution groups and plugins;
  variables are plugin inputs/outputs and vehicle parameters; the solver column shows cycle solvers or execution order.</p>`);
showDefault();
applyAutomaticCollapse(initial.rootId);
applyPreferredScale();
history=[snapshot()];historyIndex=0;
  renderAll();
  requestAnimationFrame(()=>{
    const fitted=fittedScale();
    const corrected=componentById.has(state.rootId)?Math.max(DETAIL_MIN_CELL,fitted):fitted;
    if(historyIndex===0&&history.length===1&&corrected!==state.cell){
      state.cell=corrected;
      history[0]=snapshot();
      renderAll()
  }
});
</script>
</body>
</html>
"""
