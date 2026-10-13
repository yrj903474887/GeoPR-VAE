"""Aggregate controlled results by independent base, including probe strata."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .controlled import DEFAULT_CONFIG,tasks

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    ap.add_argument('--phase',choices=['development','confirmation'],default='confirmation')
    args=ap.parse_args();items=[];seen=[]
    for path in sorted(args.run.glob('*/metrics.json')):
        rec=json.loads(path.read_text())
        if rec['technical_smoke']:raise ValueError('Technical smoke results cannot be summarized as research results')
        task=rec['task'];meta=rec['probe_cases']
        seen.append(task)
        with np.load(path.parent/'case_metrics.npz',allow_pickle=False) as z:
            matrix=z['case_metrics'];names=z['case_metric_names'].astype(str)
        items.append(dict(**task,factor='overall',level='all',values=rec['summary']))
        items.append(dict(**task,factor='noise',level=task['noise'],values=rec['summary']))
        for factor in ('amplitude','support_size','shape'):
            for level in sorted({r[factor] for r in meta}):
                keep=np.array([r[factor]==level for r in meta])
                values={str(n):float(v) for n,v in zip(names,matrix[keep].mean(0))}
                items.append(dict(**task,factor=factor,level=level,values=values))
    if not items:raise ValueError('No completed research tasks found')
    cfg=json.loads(args.config.read_text());expected={r['id'] for r in tasks(cfg,args.phase,sorted({t['method'] for t in seen}))}
    found=[r['id'] for r in seen]
    if len(found)!=len(set(found)) or set(found)!=expected:
        raise ValueError(f'Incomplete or duplicate experiment matrix: {len(set(found))} tasks found, {len(expected)} required')
    groups=defaultdict(list)
    for item in items:
        for metric,value in item['values'].items():
            if metric.startswith('n_') or metric=='calibration_threshold':continue
            key=tuple(item[k] for k in ('method','base_seed','buffer_m','factor','level'))+(metric,)
            groups[key].append(value)
    rows=[]
    keys=('method','base_seed','buffer_m','factor','level','metric')
    for key,values in groups.items():rows.append(dict(zip(keys,key),value=float(np.mean(values)),nested_tasks=len(values)))
    args.output.mkdir(parents=True,exist_ok=True);write_csv(args.output/'base_means_long.csv',rows)
    for factor,filename in [('overall','base_means.csv'),('noise','noise_base_means.csv'),('amplitude','amplitude_base_means.csv')]:
        wide={}
        for row in rows:
            if row['factor']!=factor:continue
            key=(row['method'],row['buffer_m'],row['base_seed'],row['level'])
            if key not in wide:
                wide[key]={k:row[k] for k in ('method','buffer_m','base_seed')}
                if factor!='overall':wide[key][factor]=row['level']
            wide[key][row['metric']]=row['value']
        write_csv(args.output/filename,list(wide.values()))
    pooled=defaultdict(list)
    for row in rows:pooled[tuple(row[k] for k in ('method','buffer_m','factor','level','metric'))].append(row['value'])
    summary=[]
    for key,values in pooled.items():
        v=np.asarray(values);summary.append(dict(zip(('method','buffer_m','factor','level','metric'),key),
                                               mean=float(v.mean()),sd=float(v.std(ddof=1)) if len(v)>1 else 0.,n_base=len(v)))
    write_csv(args.output/'summary.csv',summary)

if __name__=='__main__':main()
