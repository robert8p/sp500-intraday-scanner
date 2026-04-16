import { useState, useEffect, useCallback } from "react";

const SCAN_HOURS = [10,11,12,13,14,15];
const F = "'JetBrains Mono','SF Mono','Fira Code','Cascadia Code',monospace";
const Box = ({children,style})=><div style={{background:"rgba(255,255,255,0.015)",borderRadius:8,border:"1px solid rgba(255,255,255,0.04)",padding:16,...style}}>{children}</div>;
const Lbl = ({children})=><div style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase",marginBottom:10}}>{children}</div>;
const Btn = ({children,active,color="#3b82f6",onClick,disabled,style:s})=>(
  <button onClick={onClick} disabled={disabled} style={{padding:"4px 10px",borderRadius:4,fontSize:11,fontFamily:F,
    cursor:disabled?"default":"pointer",border:`1px solid ${active?color:"rgba(255,255,255,0.06)"}`,
    background:active?`${color}18`:"transparent",color:active?color:disabled?"#334155":"#64748b",
    opacity:disabled?0.5:1,transition:"all 0.15s",...s}}>{children}</button>);

function SourceBadge({source,trained}) {
  const cfg={live:{c:"#22c55e",l:"● LIVE"},cached:{c:"#eab308",l:"LAST SCAN"},offline:{c:"#64748b",l:"OFFLINE"},loading:{c:"#64748b",l:"..."},error:{c:"#ef4444",l:"ERROR"}}[source]||{c:"#64748b",l:"?"};
  return (
    <div style={{display:"flex",gap:6,alignItems:"center"}}>
      <span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:`${cfg.c}18`,color:cfg.c,border:`1px solid ${cfg.c}30`}}>{cfg.l}</span>
      {trained&&<span style={{fontSize:9,padding:"2px 8px",borderRadius:3,fontWeight:700,letterSpacing:0.5,background:"rgba(139,92,246,0.12)",color:"#8b5cf6",border:"1px solid rgba(139,92,246,0.2)"}}>LGBM FIRST-PASSAGE</span>}
    </div>);
}

function WinBar({winProb,ev}) {
  const pct=(winProb*100).toFixed(1);
  const c=winProb>0.70?"#22c55e":winProb>0.65?"#a3e635":winProb>0.61?"#eab308":winProb>0.55?"#f97316":"#6b7280";
  const evColor = ev>0?"#22c55e":ev<0?"#ef4444":"#64748b";
  return (
    <div style={{display:"flex",alignItems:"center",gap:6,minWidth:180}}>
      <div style={{width:50,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden",flexShrink:0}}>
        <div style={{width:`${Math.max(2,winProb*100)}%`,height:"100%",background:c,borderRadius:3,transition:"width 0.4s"}}/>
      </div>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:12,color:c,fontWeight:600,minWidth:40}}>{pct}%</span>
      <span style={{fontVariantNumeric:"tabular-nums",fontSize:11,color:evColor,fontWeight:500,minWidth:44}}>
        {ev>0?"+":""}{ev.toFixed(2)}%
      </span>
    </div>);
}

function Fc({value,label}) {
  const v=parseFloat(value);let c="#94a3b8";
  if(["momentum","vwapDist","vwapSlope","trendStr","orbStrength"].includes(label)) c=v>0.4?"#22c55e":v>0.15?"#a3e635":v>0?"#94a3b8":v>-0.2?"#f97316":"#ef4444";
  else if(label==="relVolume") c=v>1.8?"#22c55e":v>1.2?"#a3e635":"#94a3b8";
  else if(label==="atrReach") c=v<0.8?"#22c55e":v<1.2?"#eab308":"#ef4444";
  return <span style={{color:c,fontVariantNumeric:"tabular-nums",fontSize:11.5}}>{value}</span>;
}

// ─── SCANNER ─────────────────────────────────────────────────────
function ScannerTab({data,scanHour,source,elapsed,message,modelWR10,modelPnL10}) {
  const [mode,setMode]=useState("posEV");

  if(source==="offline"||!data||data.length===0) return (
    <Box style={{padding:40,textAlign:"center"}}>
      <div style={{fontSize:14,color:"#64748b",marginBottom:8}}>{source==="offline"?"Market closed / No model":"No data"}</div>
      <div style={{fontSize:12,color:"#475569"}}>{message||"Train model, then scan during market hours."}</div>
    </Box>);

  const filtered = mode==="be" ? data.filter(s=>s.winProb>=0.612)
    : mode==="be5" ? data.filter(s=>s.winProb>=0.662)
    : mode==="posEV" ? data.filter(s=>s.ev>0)
    : data.slice(0, mode==="top10"?10:20);
  const posEV = data.filter(s=>s.ev>0);
  const avgEV = posEV.length>0 ? posEV.reduce((s,r)=>s+r.ev,0)/posEV.length : 0;

  return (
    <div>
      <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:16,flexWrap:"wrap"}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5}}>Show</span>
          {[["posEV","+EV"],["be","Win>61%"],["be5","Win>66%"],["top10","Top 10"],["top20","Top 20"]].map(([m,l])=>
            <Btn key={m} active={mode===m} onClick={()=>setMode(m)}>{l}</Btn>)}
        </div>
        <span style={{fontSize:11,color:"#334155"}}>
          {scanHour}:00 ET — {posEV.length} positive-EV stocks
          {elapsed!=null&&` — ${elapsed}ms`}
          {modelWR10!=null&&` — val WR@10 ${(modelWR10*100).toFixed(0)}%`}
          {modelPnL10!=null&&` — val PnL@10 ${modelPnL10>0?"+":""}${modelPnL10}%`}
        </span>
      </div>

      {filtered.length===0 ? (
        <Box style={{padding:20,textAlign:"center",color:"#64748b",fontSize:12}}>
          No stocks meet the threshold at this scan hour.
        </Box>
      ) : (
        <Box style={{padding:12}}>
          <div style={{display:"flex",justifyContent:"space-between",marginBottom:8}}>
            <span style={{fontSize:11,color:"#64748b",letterSpacing:0.5,textTransform:"uppercase"}}>
              {filtered.length} stocks — TP +0.95% / SL -1.50% / Close 15:55 (break-even: 61.2% win rate)
            </span>
            {posEV.length>0&&<span style={{fontSize:11,color:"#22c55e"}}>Avg EV (positive): +{avgEV.toFixed(3)}%</span>}
          </div>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
              <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
                {["#","Ticker","Sector","Price","Chg%","Win%","EV","Mom","RelVol","VWAP%","ATR","ORB","Vol","Trend","RSI"].map(h=>(
                  <th key={h} style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10,fontWeight:500,letterSpacing:0.5,textTransform:"uppercase",whiteSpace:"nowrap"}}>{h}</th>))}
              </tr></thead>
              <tbody>{filtered.map((s,i)=>{const chg=parseFloat(s.changeFromOpen);const evPos=s.ev>0;return(
                <tr key={s.ticker+s.rank} style={{borderBottom:"1px solid rgba(255,255,255,0.03)",
                  background:evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}}
                  onMouseEnter={e=>e.currentTarget.style.background="rgba(255,255,255,0.04)"}
                  onMouseLeave={e=>e.currentTarget.style.background=evPos?"rgba(34,197,94,0.03)":i%2?"rgba(255,255,255,0.015)":"transparent"}>
                  <td style={{padding:"5px 6px",color:"#475569",fontWeight:600,fontSize:11}}>{s.rank}</td>
                  <td style={{padding:"5px 6px",fontWeight:700,color:"#e2e8f0",letterSpacing:0.3}}>{s.ticker}</td>
                  <td style={{padding:"5px 6px",color:"#64748b",fontSize:11}}>{s.sector}</td>
                  <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>${s.price}</td>
                  <td style={{padding:"5px 6px",color:chg>0?"#22c55e":chg<0?"#ef4444":"#94a3b8",fontVariantNumeric:"tabular-nums",fontWeight:500}}>{chg>0?"+":""}{chg}%</td>
                  <td style={{padding:"5px 6px"}} colSpan={2}><WinBar winProb={s.winProb} ev={s.ev}/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.momentum} label="momentum"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.relVolume} label="relVolume"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.vwapDist} label="vwapDist"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.atrReach} label="atrReach"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.orbStrength} label="orbStrength"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.realizedVol} label="realizedVol"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.trendStr} label="trendStr"/></td>
                  <td style={{padding:"5px 6px"}}><Fc value={s.features.rsi} label="rsi"/></td>
                </tr>);})}</tbody>
            </table>
          </div>
        </Box>
      )}
    </div>);
}

// ─── TRAINING ────────────────────────────────────────────────────
function TrainingTab() {
  const [d,setD]=useState(null);
  const [ld,setLd]=useState(true);
  const [sh,setSh]=useState(10);

  const poll=useCallback(()=>{fetch('/api/training/progress').then(r=>r.json()).then(d=>{setD(d);setLd(false);}).catch(()=>setLd(false));},[]);
  useEffect(()=>{poll();const iv=setInterval(poll,2000);return()=>clearInterval(iv);},[poll]);
  const trigTrain=async()=>{await fetch('/api/train',{method:'POST'});poll();};

  if(ld) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  const ip=d?.inProgress,pg=d||{},meta=d?.meta||{};
  const sm=meta[String(sh)];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Model Training — First-Passage (+0.95% TP / -1.50% SL, break-even 61.2%)</Lbl>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
          <div style={{fontSize:12,lineHeight:2,color:"#94a3b8"}}>
            {[
              {l:"Models",v:Object.keys(meta).length>0?Object.keys(meta).map(h=>h+":00").join(", "):"None",ok:Object.keys(meta).length>0},
              {l:"Trained",v:Object.values(meta)[0]?.trained_at?new Date(Object.values(meta)[0].trained_at).toLocaleDateString():"Never",ok:Object.keys(meta).length>0},
              {l:"Strategy",v:`TP +${Object.values(meta)[0]?.tp_pct||0.95}% / SL -${Object.values(meta)[0]?.sl_pct||0.95}%`,ok:true},
            ].map((c,i)=>(
              <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
                <span style={{width:14,height:14,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
                  background:c.ok?"rgba(34,197,94,0.15)":"rgba(239,68,68,0.15)",color:c.ok?"#22c55e":"#ef4444",fontSize:10,fontWeight:900}}>{c.ok?"✓":"✗"}</span>
                <span style={{minWidth:120}}>{c.l}</span>
                <span style={{color:c.ok?"#e2e8f0":"#ef4444",fontWeight:500}}>{c.v}</span>
              </div>))}
          </div>
          <div>
            <Btn onClick={trigTrain} disabled={ip} color="#8b5cf6" style={{padding:"8px 16px",fontSize:12}}>
              {ip?"Training...":"Train Models (12 months)"}
            </Btn>
            {ip&&(
              <div style={{marginTop:10}}>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{flex:1,height:6,background:"rgba(255,255,255,0.06)",borderRadius:3,overflow:"hidden"}}>
                    <div style={{width:`${pg.pct||0}%`,height:"100%",background:"#8b5cf6",borderRadius:3,transition:"width 0.5s"}}/>
                  </div>
                  <span style={{fontSize:11,color:"#8b5cf6",fontWeight:600}}>{pg.pct||0}%</span>
                </div>
                <div style={{fontSize:11,color:"#64748b"}}>{pg.message}</div>
              </div>)}
          </div>
        </div>
      </Box>

      {Object.keys(meta).length>0&&(
        <Box>
          <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
            <Lbl>Validation Results</Lbl>
            <div style={{display:"flex",gap:4,marginBottom:10}}>
              {SCAN_HOURS.filter(h=>meta[String(h)]).map(h=><Btn key={h} active={h===sh} onClick={()=>setSh(h)}>{h}:00</Btn>)}
            </div>
          </div>
          {sm?(
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Key Metrics</div>
                <div style={{fontSize:12,lineHeight:2.2,color:"#94a3b8"}}>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>AUC:</span>
                    <span style={{color:sm.auc>0.6?"#22c55e":"#eab308",fontWeight:700,fontSize:14}}>{sm.auc}</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Win Rate:</span>
                    <span style={{color:sm.avg_win_rate_top10>sm.val_win_rate?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{(sm.avg_win_rate_top10*100).toFixed(1)}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>vs base {(sm.val_win_rate*100).toFixed(1)}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Top-10 Avg P&L:</span>
                    <span style={{color:sm.avg_pnl_top10>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.avg_pnl_top10>0?"+":""}{sm.avg_pnl_top10}%</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;61% stocks):</span>
                    <span style={{color:sm.ev_above_breakeven>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven>0?"+":""}{sm.ev_above_breakeven}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven} stocks)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Win&gt;66% stocks):</span>
                    <span style={{color:sm.ev_above_breakeven_plus5>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_breakeven_plus5>0?"+":""}{sm.ev_above_breakeven_plus5}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_breakeven_plus5} stocks)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>EV (Top-10 default):</span>
                    <span style={{color:sm.ev_above_50pct>0?"#22c55e":"#ef4444",fontWeight:700,fontSize:14}}>{sm.ev_above_50pct>0?"+":""}{sm.ev_above_50pct}%</span>
                    <span style={{color:"#475569",fontSize:11,marginLeft:8}}>({sm.n_above_50pct} samples @ &gt;50%)</span></div>
                  <div><span style={{color:"#64748b",display:"inline-block",minWidth:160}}>Exit reasons (val):</span>
                    <span style={{fontSize:11}}>{sm.val_exit_reasons?Object.entries(sm.val_exit_reasons).map(([r,n])=>`${r}: ${n}`).join(", "):""}</span></div>
                </div>
              </div>
              <div>
                <div style={{fontSize:11,color:"#64748b",marginBottom:8,textTransform:"uppercase",letterSpacing:0.5}}>Feature Importance</div>
                {sm.importance&&Object.entries(sm.importance).sort(([,a],[,b])=>b-a).slice(0,12).map(([name,val])=>{
                  const max=Math.max(...Object.values(sm.importance));
                  return (
                    <div key={name} style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
                      <span style={{width:100,fontSize:10,color:"#94a3b8",textAlign:"right",flexShrink:0}}>{name}</span>
                      <div style={{flex:1,height:12,background:"rgba(255,255,255,0.04)",borderRadius:2,overflow:"hidden"}}>
                        <div style={{width:`${(val/max)*100}%`,height:"100%",borderRadius:2,background:"#8b5cf6"}}/>
                      </div>
                      <span style={{fontSize:10,color:"#64748b",minWidth:32,fontVariantNumeric:"tabular-nums"}}>{(val*100).toFixed(1)}%</span>
                    </div>);})}
              </div>
            </div>
          ):<div style={{color:"#475569",fontSize:12}}>Select scan hour</div>}
        </Box>)}
    </div>);
}

// ─── OUTCOMES ────────────────────────────────────────────────────
function OutcomesTab() {
  const [d,setD]=useState(null);
  useEffect(()=>{fetch('/api/outcomes/summary').then(r=>r.json()).then(setD).catch(()=>{});},[]);
  if(!d) return <div style={{color:"#475569",padding:40,textAlign:"center"}}>Loading...</div>;
  if(d.totalDays===0) return <Box><div style={{color:"#475569",fontSize:12,padding:20,textAlign:"center"}}>No outcomes yet. Recorded at 16:12 ET each trading day.</div></Box>;
  return (
    <div style={{display:"flex",flexDirection:"column",gap:12}}>
      <Box>
        <Lbl>Top-10 Win Rate & P&L — {d.totalDays} days</Lbl>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:12}}>
            <thead><tr style={{borderBottom:"1px solid rgba(255,255,255,0.08)"}}>
              <th style={{padding:"6px",textAlign:"left",color:"#64748b",fontSize:10}}>DATE</th>
              {SCAN_HOURS.map(h=><th key={h} style={{padding:"6px",textAlign:"center",color:"#64748b",fontSize:10}}>{h}:00</th>)}
            </tr></thead>
            <tbody>{d.recent.map((day,i)=>(
              <tr key={i} style={{borderBottom:"1px solid rgba(255,255,255,0.03)"}}>
                <td style={{padding:"5px 6px",color:"#94a3b8",fontVariantNumeric:"tabular-nums"}}>{day.date}</td>
                {SCAN_HOURS.map(h=>{
                  const s=day.hours[String(h)];
                  if(!s) return <td key={h} style={{padding:"5px 6px",textAlign:"center",color:"#334155"}}>—</td>;
                  const wr=s.top10wins*10;
                  const pnl=s.top10pnl||0;
                  const bwr=s.baseWR||0;
                  const c=wr>bwr+10?"#22c55e":wr>bwr?"#eab308":"#ef4444";
                  const pc=pnl>0?"#22c55e":"#ef4444";
                  return <td key={h} style={{padding:"5px 6px",textAlign:"center",fontVariantNumeric:"tabular-nums"}}>
                    <span style={{color:c,fontWeight:600}}>{wr}%</span>
                    <span style={{fontSize:10,color:"#475569",marginLeft:3}}>({s.top10wins}/10)</span>
                    <div style={{fontSize:10,color:pc,fontWeight:500}}>{pnl>0?"+":""}{pnl}%</div>
                    <div style={{fontSize:9,color:"#334155"}}>base {bwr}%</div>
                  </td>;
                })}
              </tr>))}</tbody>
          </table>
        </div>
      </Box>
    </div>);
}

// ─── STATUS ──────────────────────────────────────────────────────
function StatusTab({health}) {
  return <Box><Lbl>Server</Lbl>
    {health?<div style={{fontSize:12,lineHeight:2,color:"#94a3b8"}}>
      {[
        {l:"Server",v:"Online",ok:true},
        {l:"Alpaca",v:health.hasCredentials?"OK":"NOT SET",ok:health.hasCredentials},
        {l:"Market",v:health.marketOpen?"Open":"Closed",ok:health.marketOpen},
        {l:"Models",v:health.modelsLoaded?.length>0?health.modelsLoaded.join(", "):"None",ok:health.modelsLoaded?.length>0},
        {l:"Strategy",v:`TP +${health.tp_pct}% / SL -${health.sl_pct}%`,ok:true},
        {l:"Outcome days",v:String(health.outcomeDays||0),ok:(health.outcomeDays||0)>0},
        {l:"Cached scans",v:health.lastScanHours?.length>0?health.lastScanHours.join(", "):"None",ok:health.hasLastScan},
      ].map((c,i)=>(
        <div key={i} style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{width:14,height:14,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
            background:c.ok?"rgba(34,197,94,0.15)":"rgba(239,68,68,0.15)",color:c.ok?"#22c55e":"#ef4444",fontSize:10,fontWeight:900}}>{c.ok?"✓":"✗"}</span>
          <span style={{minWidth:140}}>{c.l}</span>
          <span style={{color:c.ok?"#e2e8f0":"#ef4444",fontWeight:500}}>{c.v}</span>
        </div>))}
    </div>:<div style={{color:"#ef4444"}}>Cannot reach server</div>}
  </Box>;
}

// ─── MAIN ────────────────────────────────────────────────────────
export default function SP500Scanner() {
  const [scanHour,setScanHour]=useState(10);
  const [tab,setTab]=useState("scanner");
  const [data,setData]=useState([]);
  const [source,setSource]=useState("loading");
  const [loading,setLoading]=useState(true);
  const [lastUpdate,setLastUpdate]=useState(null);
  const [elapsed,setElapsed]=useState(null);
  const [modelWR10,setModelWR10]=useState(null);
  const [modelPnL10,setModelPnL10]=useState(null);
  const [health,setHealth]=useState(null);
  const [error,setError]=useState(null);
  const [message,setMessage]=useState(null);

  useEffect(()=>{fetch('/api/health').then(r=>r.json()).then(setHealth).catch(()=>{});},[]);

  const fetchScan=useCallback(async(hour,force=false)=>{
    setLoading(true);setError(null);setMessage(null);
    try{
      const url=force?`/api/scan/${hour}/refresh`:`/api/scan/${hour}`;
      const r=await fetch(url,force?{method:'POST'}:{});
      if(!r.ok){const e=await r.json().catch(()=>({}));throw new Error(e.error||`HTTP ${r.status}`);}
      const d=await r.json();
      setData(d.data||[]);setSource(d.source||"offline");setLastUpdate(d.timestamp);
      setElapsed(d.elapsed||null);setModelWR10(d.modelWR10||null);setModelPnL10(d.modelPnL10||null);
      setMessage(d.message||null);
    }catch(err){setError(err.message);setSource("error");setData([]);}
    finally{setLoading(false);}
  },[]);

  useEffect(()=>{fetchScan(scanHour);},[scanHour,fetchScan]);
  useEffect(()=>{if(source!=="live")return;const iv=setInterval(()=>fetchScan(scanHour),5*60*1000);return()=>clearInterval(iv);},[source,scanHour,fetchScan]);

  const downloadDiag=useCallback(async()=>{
    try{const r=await fetch('/api/diagnostic');const b=await r.blob();
      const fn=r.headers.get('content-disposition')?.match(/filename="(.+)"/)?.[1]||`diag.json`;
      const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=fn;document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(u);
    }catch(e){alert(e.message);}
  },[]);

  const tabs=[{id:"scanner",l:"Scanner"},{id:"training",l:"Training",c:"#8b5cf6"},{id:"outcomes",l:"Outcomes",c:"#22c55e"},{id:"status",l:"Status"}];

  return (
    <div style={{fontFamily:F,background:"#0c0f14",color:"#e2e8f0",minHeight:"100vh"}}>
      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"12px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{fontSize:15,fontWeight:800,letterSpacing:1}}>S&P 500 SCANNER</div>
          <SourceBadge source={loading?"loading":source} trained={health?.trained}/>
        </div>
        <div style={{display:"flex",gap:4,fontSize:11,flexWrap:"wrap",alignItems:"center"}}>
          <span style={{color:"#eab308",fontWeight:600}}>TP +0.95% / SL -1.50% / Close 15:55</span>
          {lastUpdate&&<><span style={{color:"#334155",margin:"0 4px"}}>|</span><span style={{color:"#94a3b8"}}>{new Date(lastUpdate).toLocaleString()}</span></>}
        </div>
      </div>

      <div style={{borderBottom:"1px solid rgba(255,255,255,0.06)",padding:"8px 20px",display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:8}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#475569",textTransform:"uppercase",letterSpacing:0.5,marginRight:4}}>Scan</span>
          {SCAN_HOURS.map(h=><Btn key={h} active={h===scanHour} onClick={()=>setScanHour(h)}>{h}:00</Btn>)}
          <Btn onClick={()=>fetchScan(scanHour,true)} disabled={!health?.marketOpen||!health?.trained} style={{marginLeft:4}}>↻ Refresh</Btn>
          <Btn onClick={downloadDiag} color="#f97316" style={{marginLeft:4}}>⬇ Diagnostic</Btn>
        </div>
        <div style={{display:"flex",gap:2}}>
          {tabs.map(t=><Btn key={t.id} active={t.id===tab} onClick={()=>setTab(t.id)} color={t.c||"#3b82f6"}>{t.l}</Btn>)}
        </div>
      </div>

      {error&&<div style={{margin:"12px 20px 0",padding:"8px 12px",borderRadius:6,background:"rgba(239,68,68,0.1)",border:"1px solid rgba(239,68,68,0.2)",color:"#ef4444",fontSize:12}}>{error}</div>}

      <div style={{padding:"16px 20px"}}>
        {tab==="scanner"&&<ScannerTab data={data} scanHour={scanHour} source={source} elapsed={elapsed} message={message} modelWR10={modelWR10} modelPnL10={modelPnL10}/>}
        {tab==="training"&&<TrainingTab/>}
        {tab==="outcomes"&&<OutcomesTab/>}
        {tab==="status"&&<StatusTab health={health}/>}
      </div>
    </div>);
}
