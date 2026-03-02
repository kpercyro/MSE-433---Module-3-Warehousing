import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const SHAPES = ['circle','pentagon','trapezoid','triangle','star','moon','heart','cross'];
const SHAPE_EMOJI = ['⬤','⬠','⏢','△','★','☽','♥','✚'];
const BELT_COLORS = ['#E63946','#457B9D','#2A9D8F','#E9C46A'];
const SHAPE_COLORS = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51','#606C88','#E63946','#457B9D'];

const DATA = {
  orders: [
    {id:0,items:[{type:'pentagon',tid:1,qty:2},{type:'triangle',tid:3,qty:3}],total:5,totes:[0,0],ct:206},
    {id:1,items:[{type:'trapezoid',tid:2,qty:1},{type:'triangle',tid:3,qty:3},{type:'star',tid:4,qty:1}],total:5,totes:[14,4,14],ct:336},
    {id:2,items:[{type:'moon',tid:5,qty:2}],total:2,totes:[7],ct:130},
    {id:3,items:[{type:'circle',tid:0,qty:3},{type:'moon',tid:5,qty:1}],total:4,totes:[9,10],ct:374},
    {id:4,items:[{type:'pentagon',tid:1,qty:1},{type:'trapezoid',tid:2,qty:1}],total:2,totes:[11,12],ct:108},
    {id:5,items:[{type:'pentagon',tid:1,qty:2}],total:2,totes:[1],ct:42},
    {id:6,items:[{type:'triangle',tid:3,qty:2},{type:'moon',tid:5,qty:1}],total:3,totes:[3,13],ct:86},
    {id:7,items:[{type:'trapezoid',tid:2,qty:1},{type:'star',tid:4,qty:2}],total:3,totes:[8,8],ct:160},
    {id:8,items:[{type:'circle',tid:0,qty:3},{type:'pentagon',tid:1,qty:2},{type:'moon',tid:5,qty:1}],total:6,totes:[13,8,13],ct:290},
    {id:9,items:[{type:'heart',tid:6,qty:3}],total:3,totes:[1],ct:236},
    {id:10,items:[{type:'pentagon',tid:1,qty:1}],total:1,totes:[10],ct:56},
  ],
  belts: [
    {id:0,queue:[5,2,8]},
    {id:1,queue:[10,7,1]},
    {id:2,queue:[6,0]},
    {id:3,queue:[4,9,3]},
  ],
  naive_belts: [{queue:[0,4,8]},{queue:[1,5,9]},{queue:[2,6,10]},{queue:[3,7]}],
  metrics: {makespan:374,total_ct:1810,avg_ct:164.5,naive_makespan:446},
  completion: [[5,42],[10,56],[6,86],[4,108],[2,130],[7,160],[0,206],[9,236],[8,290],[1,336],[3,374]],
};

const ShapeIcon = ({type, size=16}) => {
  const idx = SHAPES.indexOf(type);
  return <span style={{fontSize:size,color:SHAPE_COLORS[idx]||'#666',lineHeight:1}}>{SHAPE_EMOJI[idx]||'?'}</span>;
};

const Metric = ({label,value,unit,accent}) => (
  <div style={{textAlign:'center',padding:'12px 16px',background:accent?'#1a1a2e':'#16213e',borderRadius:8,border:accent?'1px solid #E63946':'1px solid #1a1a2e'}}>
    <div style={{fontSize:11,color:'#8892b0',textTransform:'uppercase',letterSpacing:1,marginBottom:4}}>{label}</div>
    <div style={{fontSize:accent?28:22,fontWeight:700,color:accent?'#E63946':'#ccd6f6'}}>{value}<span style={{fontSize:13,fontWeight:400,color:'#8892b0',marginLeft:3}}>{unit}</span></div>
  </div>
);

const BeltViz = ({belt,orders}) => {
  const total = belt.queue.reduce((s,oi)=>s+orders[oi].total,0);
  return (
    <div style={{background:'#1a1a2e',borderRadius:10,padding:14,border:`2px solid ${BELT_COLORS[belt.id]}33`}}>
      <div style={{display:'flex',alignItems:'center',gap:8,marginBottom:10}}>
        <div style={{width:12,height:12,borderRadius:3,background:BELT_COLORS[belt.id]}}/>
        <span style={{fontWeight:700,color:BELT_COLORS[belt.id],fontSize:14}}>Belt {belt.id}</span>
        <span style={{fontSize:12,color:'#8892b0',marginLeft:'auto'}}>{total} items</span>
      </div>
      <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
        {belt.queue.map((oi,idx)=>{
          const o = orders[oi];
          return (
            <div key={idx} style={{background:'#16213e',borderRadius:6,padding:'8px 10px',border:'1px solid #233554',flex:'1 1 auto',minWidth:80}}>
              <div style={{fontSize:11,color:'#8892b0',marginBottom:4}}>
                {idx===0?'▶ Active':idx===1?'Next':'Queued'}
              </div>
              <div style={{fontWeight:600,color:'#ccd6f6',fontSize:13,marginBottom:4}}>Order {oi}</div>
              <div style={{display:'flex',gap:4,flexWrap:'wrap'}}>
                {o.items.map((it,j)=>(
                  <span key={j} style={{display:'inline-flex',alignItems:'center',gap:2,fontSize:12,color:'#a8b2d1'}}>
                    {it.qty}×<ShapeIcon type={it.type} size={14}/>
                  </span>
                ))}
              </div>
              <div style={{fontSize:11,color:BELT_COLORS[belt.id],marginTop:4}}>
                {(o.ct/60).toFixed(1)}min
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default function App() {
  const [tab, setTab] = useState('overview');
  const {orders, belts, metrics, completion} = DATA;
  
  const chartData = completion.map(([oi,ct])=>({
    name:`O${oi}`,order:oi,time:ct,items:orders[oi].total,
    belt:belts.findIndex(b=>b.queue.includes(oi))
  }));
  
  const beltLoadData = belts.map(b=>({
    name:`Belt ${b.id}`,
    items:b.queue.reduce((s,oi)=>s+orders[oi].total,0),
    orders:b.queue.length
  }));

  const tabs = [
    {key:'overview',label:'Overview'},
    {key:'belts',label:'Belt Assignment'},
    {key:'timeline',label:'Timeline'},
    {key:'loading',label:'Loading Plan'},
    {key:'algorithm',label:'Algorithm'},
  ];

  return (
    <div style={{fontFamily:'"JetBrains Mono",monospace',background:'#0a192f',color:'#ccd6f6',minHeight:'100vh',padding:20}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');`}</style>
      
      <div style={{maxWidth:900,margin:'0 auto'}}>
        <div style={{marginBottom:24,borderBottom:'1px solid #233554',paddingBottom:16}}>
          <h1 style={{fontSize:20,fontWeight:700,color:'#E63946',margin:0,letterSpacing:1}}>
            MSE433 CONVEYOR OPTIMIZER
          </h1>
          <p style={{fontSize:12,color:'#8892b0',margin:'4px 0 0'}}>
            Simulated Annealing · Zhou (2017) · Seed 100 · 11 orders · 36 items
          </p>
        </div>

        <div style={{display:'flex',gap:6,marginBottom:20,flexWrap:'wrap'}}>
          {tabs.map(t=>(
            <button key={t.key} onClick={()=>setTab(t.key)} style={{
              padding:'6px 14px',borderRadius:6,border:'none',cursor:'pointer',fontSize:12,fontWeight:600,
              fontFamily:'inherit',letterSpacing:0.5,
              background:tab===t.key?'#E63946':'#1a1a2e',
              color:tab===t.key?'#fff':'#8892b0',
            }}>{t.label}</button>
          ))}
        </div>

        {tab==='overview' && (
          <div>
            <div style={{display:'grid',gridTemplateColumns:'repeat(4,1fr)',gap:10,marginBottom:20}}>
              <Metric label="Makespan" value={(metrics.makespan/60).toFixed(1)} unit="min" accent/>
              <Metric label="vs Naive" value={`-${((1-metrics.makespan/metrics.naive_makespan)*100).toFixed(0)}%`} unit="faster"/>
              <Metric label="Avg Completion" value={(metrics.avg_ct/60).toFixed(1)} unit="min"/>
              <Metric label="Total Items" value="36" unit="items"/>
            </div>
            
            <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14}}>
              <div style={{background:'#1a1a2e',borderRadius:10,padding:16}}>
                <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 12px',letterSpacing:1}}>ORDER COMPLETION</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={chartData} margin={{top:5,right:5,bottom:5,left:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#233554"/>
                    <XAxis dataKey="name" tick={{fill:'#8892b0',fontSize:11}} stroke="#233554"/>
                    <YAxis tick={{fill:'#8892b0',fontSize:11}} stroke="#233554" label={{value:'sec',angle:-90,position:'insideLeft',fill:'#8892b0',fontSize:11}}/>
                    <Tooltip contentStyle={{background:'#16213e',border:'1px solid #233554',borderRadius:6,fontSize:12,color:'#ccd6f6'}}
                      formatter={(v)=>[`${v}s (${(v/60).toFixed(1)}min)`,'Completion']}/>
                    <Bar dataKey="time" radius={[4,4,0,0]}>
                      {chartData.map((d,i)=><Cell key={i} fill={BELT_COLORS[d.belt]}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              
              <div style={{background:'#1a1a2e',borderRadius:10,padding:16}}>
                <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 12px',letterSpacing:1}}>BELT WORKLOAD</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={beltLoadData} margin={{top:5,right:5,bottom:5,left:5}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#233554"/>
                    <XAxis dataKey="name" tick={{fill:'#8892b0',fontSize:11}} stroke="#233554"/>
                    <YAxis tick={{fill:'#8892b0',fontSize:11}} stroke="#233554"/>
                    <Tooltip contentStyle={{background:'#16213e',border:'1px solid #233554',borderRadius:6,fontSize:12,color:'#ccd6f6'}}/>
                    <Bar dataKey="items" name="Items" radius={[4,4,0,0]}>
                      {beltLoadData.map((d,i)=><Cell key={i} fill={BELT_COLORS[i]}/>)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div style={{background:'#1a1a2e',borderRadius:10,padding:16,marginTop:14}}>
              <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 10px',letterSpacing:1}}>ALL ORDERS</h3>
              <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(180px,1fr))',gap:8}}>
                {orders.map(o=>(
                  <div key={o.id} style={{background:'#16213e',borderRadius:6,padding:10,border:'1px solid #233554'}}>
                    <div style={{display:'flex',justifyContent:'space-between',marginBottom:6}}>
                      <span style={{fontWeight:700,fontSize:13}}>Order {o.id}</span>
                      <span style={{fontSize:11,color:'#8892b0'}}>{o.total} items</span>
                    </div>
                    <div style={{display:'flex',gap:6,flexWrap:'wrap',marginBottom:6}}>
                      {o.items.map((it,j)=>(
                        <span key={j} style={{display:'flex',alignItems:'center',gap:3,background:'#0a192f',padding:'2px 6px',borderRadius:4,fontSize:12}}>
                          {it.qty}×<ShapeIcon type={it.type} size={13}/>
                        </span>
                      ))}
                    </div>
                    <div style={{fontSize:11,color:'#8892b0'}}>
                      Totes: {o.totes.join(', ')} · Done @ {(o.ct/60).toFixed(1)}min
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {tab==='belts' && (
          <div style={{display:'flex',flexDirection:'column',gap:12}}>
            <div style={{background:'#16213e',borderRadius:8,padding:12,fontSize:13,color:'#a8b2d1',lineHeight:1.6}}>
              <strong style={{color:'#E63946'}}>Strategy:</strong> LPT (Longest Processing Time) assigns largest orders first to least-loaded belt. 
              SPT (Shortest Processing Time) orders each belt's queue smallest-first. 
              SA (Simulated Annealing) refines with 20,000 iterations using insertion + swap neighborhoods.
              Result: balanced loads of 10, 9, 8, 9 items across belts — <strong style={{color:'#2A9D8F'}}>10.8% faster</strong> than naive round-robin.
            </div>
            {belts.map(b=><BeltViz key={b.id} belt={b} orders={orders}/>)}
          </div>
        )}

        {tab==='timeline' && (
          <div style={{background:'#1a1a2e',borderRadius:10,padding:20}}>
            <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 16px',letterSpacing:1}}>GANTT TIMELINE</h3>
            <div style={{position:'relative',paddingLeft:60}}>
              {belts.map(b=>{
                let cumStart = 0;
                return (
                  <div key={b.id} style={{display:'flex',alignItems:'center',marginBottom:12,height:36}}>
                    <div style={{position:'absolute',left:0,width:55,fontSize:12,color:BELT_COLORS[b.id],fontWeight:700}}>
                      Belt {b.id}
                    </div>
                    <div style={{flex:1,position:'relative',height:'100%',background:'#16213e',borderRadius:6,overflow:'hidden'}}>
                      {b.queue.map((oi,idx)=>{
                        const o = orders[oi];
                        const prevEnd = idx > 0 ? orders[b.queue[idx-1]].ct : 0;
                        const width = ((o.ct - prevEnd) / metrics.makespan) * 100;
                        const left = (prevEnd / metrics.makespan) * 100;
                        return (
                          <div key={idx} style={{
                            position:'absolute',left:`${left}%`,width:`${width}%`,height:'100%',
                            background:`${BELT_COLORS[b.id]}${idx===0?'':'99'}`,
                            display:'flex',alignItems:'center',justifyContent:'center',
                            fontSize:11,fontWeight:600,color:'#fff',borderRight:'1px solid #0a192f',
                            overflow:'hidden',whiteSpace:'nowrap'
                          }}>
                            O{oi} ({o.total})
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
              <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'#8892b0',marginTop:4}}>
                <span>0s</span>
                <span>{(metrics.makespan/4).toFixed(0)}s</span>
                <span>{(metrics.makespan/2).toFixed(0)}s</span>
                <span>{(metrics.makespan*3/4).toFixed(0)}s</span>
                <span>{metrics.makespan}s</span>
              </div>
            </div>

            <h3 style={{fontSize:13,color:'#E63946',margin:'24px 0 12px',letterSpacing:1}}>COMPLETION SEQUENCE</h3>
            <div style={{display:'flex',flexWrap:'wrap',gap:6}}>
              {completion.map(([oi,ct],i)=>{
                const belt = belts.findIndex(b=>b.queue.includes(oi));
                return (
                  <div key={i} style={{display:'flex',alignItems:'center',gap:4}}>
                    <div style={{background:BELT_COLORS[belt],borderRadius:4,padding:'4px 8px',fontSize:12,fontWeight:600,color:'#fff'}}>
                      O{oi}
                    </div>
                    <span style={{fontSize:11,color:'#8892b0'}}>{(ct/60).toFixed(1)}m</span>
                    {i < completion.length-1 && <span style={{color:'#233554',fontSize:16}}>→</span>}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {tab==='loading' && (
          <div>
            <div style={{background:'#16213e',borderRadius:8,padding:12,fontSize:13,color:'#a8b2d1',marginBottom:14,lineHeight:1.6}}>
              <strong style={{color:'#E63946'}}>Loading Strategy:</strong> Load all items for Belt 0 first (grouped), then Belt 1, etc.
              This minimizes belt-switching delays since consecutive items for the same belt 
              sort ~8s apart vs ~14s when switching. Tested 4 strategies; sequential grouping by belt wins.
            </div>

            <div style={{background:'#1a1a2e',borderRadius:10,padding:16}}>
              <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 12px',letterSpacing:1}}>LOADING SEQUENCE (place on ramp in this order)</h3>
              {belts.map(b=>{
                if (!b.queue.length) return null;
                const oi = b.queue[0];
                const o = orders[oi];
                let step = belts.slice(0,b.id).reduce((s,bb)=>s+orders[bb.queue[0]].total,0);
                return (
                  <div key={b.id} style={{marginBottom:12}}>
                    <div style={{fontSize:12,fontWeight:700,color:BELT_COLORS[b.id],marginBottom:6}}>
                      ▸ Belt {b.id} — Order {oi}
                    </div>
                    <div style={{display:'flex',gap:4,flexWrap:'wrap',paddingLeft:12}}>
                      {o.items.flatMap((it,j)=>
                        Array.from({length:it.qty},(_,k)=>{
                          step++;
                          return (
                            <div key={`${j}-${k}`} style={{
                              background:'#16213e',border:'1px solid #233554',borderRadius:6,
                              padding:'6px 10px',display:'flex',alignItems:'center',gap:6,fontSize:12
                            }}>
                              <span style={{color:'#8892b0',fontWeight:700,fontSize:10}}>#{step}</span>
                              <ShapeIcon type={it.type} size={16}/>
                              <span style={{color:'#a8b2d1'}}>{it.type}</span>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            <div style={{background:'#1a1a2e',borderRadius:10,padding:16,marginTop:14}}>
              <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 12px',letterSpacing:1}}>SMALL TEST INSTANCE (for physical demo)</h3>
              <p style={{fontSize:12,color:'#8892b0',margin:'0 0 10px'}}>4 orders, 8 items — estimated ~2.2 min on physical conveyor</p>
              {[{b:0,o:5,items:'2× ⬠ pentagon'},{b:1,o:10,items:'1× ⬠ pentagon'},{b:2,o:6,items:'2× △ triangle, 1× ☽ moon'},{b:3,o:4,items:'1× ⬠ pentagon, 1× ⏢ trapezoid'}].map(({b,o,items})=>(
                <div key={b} style={{display:'flex',alignItems:'center',gap:10,marginBottom:6,fontSize:13}}>
                  <div style={{width:8,height:8,borderRadius:2,background:BELT_COLORS[b]}}/>
                  <span style={{color:BELT_COLORS[b],fontWeight:600,width:50}}>Belt {b}</span>
                  <span style={{color:'#ccd6f6'}}>Order {o}:</span>
                  <span style={{color:'#8892b0'}}>{items}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab==='algorithm' && (
          <div style={{display:'flex',flexDirection:'column',gap:14}}>
            <div style={{background:'#1a1a2e',borderRadius:10,padding:20}}>
              <h3 style={{fontSize:14,color:'#E63946',margin:'0 0 14px',letterSpacing:1}}>APPROACH: SIMULATION OPTIMIZATION</h3>
              <p style={{fontSize:13,color:'#a8b2d1',lineHeight:1.7,margin:0}}>
                Based on Zhou (2017), "Optimizing Order Consolidation with Simulation Optimization," 
                we model the 4-belt conveyor as a <strong style={{color:'#ccd6f6'}}>parallel machine scheduling</strong> problem.
                Each belt is a "machine," each order is a "job." The problem is NP-hard (Du et al., 1990),
                so we use a metaheuristic approach.
              </p>
            </div>
            
            {[
              {title:'1. DATA GENERATION',desc:'Random instance with configurable seed. Each order has 1-3 item types, 1-3 qty each, assigned to random totes. Items are interchangeable by type across totes.',color:'#2A9D8F'},
              {title:'2. BELT ASSIGNMENT (SA)',desc:'Simulated Annealing with LPT+SPT initial solution. Neighborhoods: insertion (50%), swap (30%), reversal (20%). Exponential cooling: T₀=100, α=0.9997, 20K iterations. Minimizes total order completion time.',color:'#457B9D'},
              {title:'3. LOADING SEQUENCE',desc:'4 strategies tested: naive sequential, round-robin interleave, belt-grouped SPT, SPT interleave. Sequential belt grouping wins — items for same belt sort faster (~8s vs ~14s switching).',color:'#E9C46A'},
              {title:'4. SIMULATION',desc:'Discrete event simulation calibrated from example output (15 items in 213s on physical conveyor). Models item loading, belt scanning, claiming, recirculation, and order completion cascading.',color:'#E76F51'},
            ].map(({title,desc,color})=>(
              <div key={title} style={{background:'#16213e',borderRadius:8,padding:14,borderLeft:`3px solid ${color}`}}>
                <h4 style={{fontSize:12,color,margin:'0 0 6px',letterSpacing:1}}>{title}</h4>
                <p style={{fontSize:12,color:'#a8b2d1',lineHeight:1.6,margin:0}}>{desc}</p>
              </div>
            ))}

            <div style={{background:'#1a1a2e',borderRadius:10,padding:16}}>
              <h3 style={{fontSize:13,color:'#E63946',margin:'0 0 10px',letterSpacing:1}}>KEY RESULTS</h3>
              <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:10}}>
                <div style={{background:'#16213e',padding:12,borderRadius:6}}>
                  <div style={{fontSize:11,color:'#8892b0',marginBottom:4}}>Naive (round-robin)</div>
                  <div style={{fontSize:18,fontWeight:700,color:'#8892b0'}}>7.4 min</div>
                </div>
                <div style={{background:'#16213e',padding:12,borderRadius:6,border:'1px solid #2A9D8F'}}>
                  <div style={{fontSize:11,color:'#2A9D8F',marginBottom:4}}>Optimized (SA + loading)</div>
                  <div style={{fontSize:18,fontWeight:700,color:'#2A9D8F'}}>6.2 min</div>
                </div>
              </div>
              <div style={{fontSize:12,color:'#a8b2d1',marginTop:10,lineHeight:1.6}}>
                <strong style={{color:'#E63946'}}>10.8% improvement</strong> in makespan. Belt loads balanced at 10/9/8/9 items.
                Small orders (O5, O10) complete in &lt;1min; largest order (O8, 6 items) at 4.8min.
                Physical conveyor bottleneck (~14s/item sort) dominates — optimization eliminates wasted recirculation and idle time.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
