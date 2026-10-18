"""Index actual registered implementations without claiming scored completion.

Different mechanism tags are a bookkeeping check, not a proof of mathematical
novelty. The hypothesis/equation and risk remain necessary reading before a
new cycle: a new name or parameter value is not a new intervention.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .registry import candidates
from .run import digest,save


def catalogue(through_cycle):
    if through_cycle<1:raise ValueError('At least one registered cycle required')
    methods={};rosters={};lineage={}
    for cycle in range(1,through_cycle+1):
        active=candidates(cycle)
        if len(active)!=20:raise ValueError('Every registered cycle needs20 candidates')
        old=set(methods);new=set(active)-old;retained=set(active)&old
        if cycle==1:
            if len(new)!=20:raise ValueError('The initial roster needs20 distinct implementations')
        else:
            if len(new)!=15 or len(retained)!=5:raise ValueError('A later roster needs five retained models and15 new children')
            if not retained<=set(rosters[str(cycle-1)]):raise ValueError('A previously dropped model was reintroduced instead of a previous-cycle survivor')
            parents=Counter(active[k].parent for k in new)
            if dict(parents)!={k:3 for k in retained}:raise ValueError('Each retained model needs three distinct children')
        rosters[str(cycle)]=list(active)
        lineage[str(cycle)]=dict(retained=sorted(retained),introduced=sorted(new),
            registered_only=True,completed_or_selected_not_inferred=True)
        for key,value in active.items():
            specification=asdict(value)
            if key in methods:
                if methods[key]['specification']!=specification:raise ValueError('A retained model changed its registered specification')
                methods[key]['present_in_cycles'].append(cycle)
            else:
                parent=value.parent
                if parent is not None and parent not in old:raise ValueError('A new child must descend from an earlier registered model')
                root=methods[parent]['lineage_root'] if parent is not None else key
                methods[key]=dict(introduced_cycle=cycle,present_in_cycles=[cycle],lineage_root=root,specification=specification)
    tags=[v['specification']['mechanism'] for v in methods.values()]
    if len(tags)!=len(set(tags)):raise ValueError('Different model IDs reused an existing mechanism tag')
    folder=Path(__file__).parent
    sources=[Path(__file__),folder/'registry.py']+sorted(folder.glob('*_manifest.py'))
    return dict(registered_cycles=through_cycle,unique_registered_models=len(methods),
        cycle_rosters=rosters,cycle_lineage=lineage,models=methods,
        source_sha256={p.name:digest(p) for p in sources},scores_read=False,predictions_read=False,
        visual_reviews_created=False,selection_created=False,
        interpretation='Descriptive index of registered code and stated hypotheses. It does not certify mathematical novelty, successful execution, convergence, biological validity, full-benchmark evaluation, or cycle completion. Compare equations and risks, not just names. Controls are outside the20 selectable candidates.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through-cycle',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=catalogue(args.through_cycle);save(args.output,result)
    print('Registered cycles:',result['registered_cycles'],'Model IDs:',result['unique_registered_models'],'No scores or predictions read')


if __name__=='__main__':main()
