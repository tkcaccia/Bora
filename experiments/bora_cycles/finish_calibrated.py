"""Finish reports/crop exports after one identified live run, then await review.

Never restarts inference, selects methods, fabricates visual reviews, or advances
the cycle. A missing/replaced process without a completed freeze is a failure.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .run import digest,save


def process_identity(pid,proc_root=Path('/proc')):
    try:tail=(proc_root/str(pid)/'stat').read_text().rpartition(')')[2].split()
    except FileNotFoundError:return None
    if len(tail)<20:raise ValueError('Malformed live-process identity')
    if tail[0] in ('Z','X'):return None
    return tail[19]


def stage_ready(root,protocol_sha256):
    root=Path(root)
    if digest(root/'PROTOCOL.json')!=protocol_sha256:raise ValueError('Identified run protocol changed')
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    if protocol['pilot']:raise ValueError('The completion pipeline requires a full run')
    stage=root/'STAGE.json'
    if not stage.exists():return False
    state=json.loads(stage.read_text())
    if state['state']!='awaiting_visual_review' or state['cases']!=len(protocol['cases']):
        raise ValueError('Unexpected full-run terminal state')
    frozen=json.loads((root/'ALL_PREDICTIONS_FROZEN.json').read_text())
    visual=json.loads((root/'VISUAL_MANIFEST.json').read_text())
    result=json.loads((root/'RESULTS.json').read_text())
    if (frozen['protocol_sha256']!=protocol_sha256 or frozen['fine_reference_read']
            or set(frozen['manifests'])!=set(protocol['cases'])
            or set(visual['contact_sha256'])!=set(protocol['cases'])
            or result['protocol']!=protocol or len(result['cases'])!=len(protocol['cases'])
            or {r['case'] for r in result['cases']}!=set(protocol['cases'])):
        raise ValueError('Terminal stage lacks the complete matching prediction/score/render freeze')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','root','previous','pilot','executed-code','crop-plan'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--start-ticks',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args();root=args.root
    journal=root/'AUDIT_PIPELINE.json'
    if journal.exists() or (root/'AUDIT_PIPELINE_COMPLETE.json').exists():raise ValueError('Preserve existing pipeline evidence')
    record=dict(state='waiting_for_identified_run',pid=args.pid,start_ticks=args.start_ticks,
        protocol_sha256=args.protocol_sha256,source_sha256=digest(Path(__file__)),steps=[],
        visual_reviews_created=False,survivors_selected=False,predictions_modified=False,scores_modified=False)
    save(journal,record)
    try:
        while not stage_ready(root,args.protocol_sha256):
            if process_identity(args.pid)!=args.start_ticks:
                raise ValueError('Identified inference process is terminal/missing/replaced without its completed freeze; do not restart automatically')
            time.sleep(30)
        env=os.environ.copy();env.update(PYTHONPATH='src:.',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            MKL_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
        steps=[('posthoc_calibrated',['--source',str(args.source),'--root',str(root),'--previous',str(args.previous),
                '--executed-code',str(args.executed_code)]),
            ('audit_repeatability',['--full',str(root),'--pilot',str(args.pilot),'--output',str(root/'PILOT_FULL_REPEATABILITY.json')]),
            ('export_planned_crops',['--source',str(args.source),'--root',str(root),'--plan',str(args.crop_plan)])]
        for name,options in steps:
            record.update(state='running_'+name,at_utc=datetime.now(timezone.utc).isoformat());save(journal,record)
            command=['nice','-n','12',sys.executable,'-u','-m','experiments.bora_cycles.'+name]+options
            print(json.dumps(dict(step=name,command=command)),flush=True)
            started=time.monotonic()
            subprocess.run(command,env=env,check=True)
            record['steps'].append(dict(step=name,seconds=time.monotonic()-started,exit_code=0))
        files=['POSTHOC_COMPLETE.json','PILOT_FULL_REPEATABILITY.json','NATIVE_CROPS_EXPORTED.json']
        record.update(state='awaiting_actual_visual_review',at_utc=datetime.now(timezone.utc).isoformat(),
            output_sha256={n:digest(root/n) for n in files},required_overview_reviews=len(json.loads((root/'PROTOCOL.json').read_text())['cases']))
        save(journal,record);save(root/'AUDIT_PIPELINE_COMPLETE.json',record)
        print(json.dumps(dict(state=record['state'],visual_reviews_created=False,survivors_selected=False)),flush=True)
    except BaseException as exc:
        record.update(state='failed',exception=repr(exc),at_utc=datetime.now(timezone.utc).isoformat())
        save(journal,record);raise


if __name__=='__main__':main()
