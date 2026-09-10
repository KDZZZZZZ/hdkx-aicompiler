#!/usr/bin/env python3
"""Audit the native CUDA adaptive hot-swap receipt (stdlib only)."""
from __future__ import annotations
import argparse, copy, json
from collections import Counter
from pathlib import Path

def require(ok, message):
    if not ok: raise AssertionError(message)

def load(path):
    manifest=json.loads((path/'manifest.json').read_text())
    require(manifest.get('enable_cupti') is True and manifest.get('cupti_available') is True, 'CUPTI was not enabled')
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    require(all(e.get('session_id')==manifest['session_id'] and e.get('trace_id')==manifest['trace_id'] for e in events), 'bundle identity mismatch')
    return events

def validate(events, generation, expected_ok, expected_error):
    runs=[e for e in events if e.get('event_type')=='runtime_session_run']
    ok=[e for e in runs if e.get('status')=='ok']
    errors=[e for e in runs if e.get('status')=='error']
    require(len(ok)==expected_ok and len(errors)==expected_error, 'run count changed')
    require(all(e.get('phase')=='complete' and e.get('device')=='cuda:0' for e in ok), 'run is not complete on CUDA')
    require(all(e.get('fields',{}).get('stage')=='adaptive_static' and e['fields'].get('generation')==str(generation) and
                e['fields'].get('adaptive_contract')=='7' and e['metrics'].get('submit_count')==3 and
                e['metrics'].get('kernel_count')==3 for e in ok), 'generation or run receipt changed')
    spans={e.get('span_id'):e for e in events if e.get('span_id')}
    for run in ok:
        rows=[e for e in events if e.get('run_id')==run['run_id'] and e.get('trace_id')==run['trace_id']]
        submits=[e for e in rows if e.get('event_type')=='kernel_submit']
        completes=[e for e in rows if e.get('event_type')=='kernel_exec']
        launches=[e for e in rows if e.get('event_type')=='kernel_launch']
        device=[e for e in rows if e.get('event_type')=='cuda_kernel']
        require(len(submits)==len(completes)==len(launches)==len(device)==3, 'kernel/device count changed')
        require({e['fields']['call_index'] for e in submits}=={'0','1','2'} and
                {e['fields']['call_index'] for e in completes}=={'0','1','2'}, 'call coverage changed')
        require(all(e.get('status')=='ok' and e.get('parent_span_id')==run.get('span_id') for e in submits+completes), 'host event parent changed')
        require(all(e.get('status')=='ok' and e.get('device')=='cuda:0' and e.get('parent_span_id') in {x.get('span_id') for x in launches} and e.get('duration_ns',0)>0 for e in device), 'CUPTI association changed')
        require({e['fields'].get('backend.cuda.stream_id') for e in device}, 'missing CUDA streams')
        for e in submits+completes+launches:
            f=e.get('fields',{})
            for key in ('generation','dispatch_key','plan_abi','plan_variant','validation_receipt','adaptive_contract'):
                require(key in f and f[key]==run['fields'].get(key), 'event lost '+key)
    for run in errors:
        rows=[e for e in events if e.get('run_id')==run['run_id']]
        require(run.get('metrics',{}).get('submit_count')==0 and not any(e.get('event_type') in {'kernel_submit','kernel_launch','kernel_exec','cuda_kernel'} for e in rows), 'rejected run launched work')
    return {"ok_runs":len(ok),"error_runs":len(errors),"kernels":sum(3 for _ in ok),"streams":sorted({e['fields']['backend.cuda.stream_id'] for e in events if e.get('event_type')=='cuda_kernel'})}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    first=load(args.root/'adaptive-first'); second=load(args.root/'adaptive-second'); reference=load(args.root/'adaptive-reference')
    a=validate(first,1,2,1); b=validate(second,2,1,0)
    first_run=next(e for e in first if e.get('event_type')=='runtime_session_run' and e.get('status')=='ok')
    second_run=next(e for e in second if e.get('event_type')=='runtime_session_run' and e.get('status')=='ok')
    require(first_run['fields']['plan_abi']==second_run['fields']['plan_abi'] and first_run['fields']['dispatch_key']==second_run['fields']['dispatch_key'], 'route/ABI changed across replacement')
    require(first_run['fields']['plan_variant']!=second_run['fields']['plan_variant'], 'selected artifact did not change')
    receipt=args.root/'adaptive-second'/'artifacts'/'health-evidence.tsv'; require(receipt.exists(), 'health receipt was not flushed')
    require(receipt.read_text().splitlines()[-1].split('\t')[:2]==['2','3'], 'health receipt does not name generation 2')
    # Exercise the same validator against tampered evidence; every mutation must fail closed.
    mutations=[]
    for kind in ('generation','missing_kernel','missing_submit','foreign_parent','device_clock','error_submit'):
        damaged=copy.deepcopy(second)
        run=next(e for e in damaged if e.get('event_type')=='runtime_session_run' and e.get('status')=='ok')
        rows=[e for e in damaged if e.get('run_id')==run['run_id']]
        if kind=='generation': run['fields']['generation']='99'
        elif kind=='missing_kernel': damaged.remove(next(e for e in rows if e.get('event_type')=='cuda_kernel'))
        elif kind=='missing_submit': damaged.remove(next(e for e in rows if e.get('event_type')=='kernel_submit'))
        elif kind=='foreign_parent': next(e for e in rows if e.get('event_type')=='cuda_kernel')['parent_span_id']='missing'
        elif kind=='device_clock': next(e for e in rows if e.get('event_type')=='cuda_kernel')['duration_ns']=0
        else: run['metrics']['submit_count']=1
        try: validate(damaged,2,1,0)
        except (AssertionError,KeyError): mutations.append(kind)
    require(len(mutations)==6, 'tamper gate accepted evidence')
    result={"backend":"cuda:0","generations":[1,2,1],"first":a,"second":b,"route":first_run['fields']['dispatch_key'],"plan_abi":first_run['fields']['plan_abi'],"negative_checks":len(mutations),"compile_reference_bundle":reference is not None}
    (args.root/'adaptive-cuda-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True)); print('[PASS] cuda_adaptive_hot_swap_profile_verified')
if __name__=='__main__': main()
