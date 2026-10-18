#!/usr/bin/env python3
"""Entry point for original tissue dataset acquisition and40-field preparation.

Run from the repository root. Dataset-specific routines, pinned source catalogs,
range-read receipts and eligibility rules live in experiments/bora_extension.
No encoder, RGB conversion, Bora inference or scoring is invoked here.
"""
import argparse
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['download-primary','download-extra','inventory','orca-download','prepare','viewer'])
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--dataset',choices=['ignite','peso','gcss','orca','puma','histoseg','gleasonxai'])
    a=p.parse_args();data=str(a.data.resolve())
    if a.action.startswith('download'):
        catalog='TISSUE_CATALOG_2026_10_18.json' if a.action=='download-primary' else 'TISSUE_CATALOG_EXTRA_SUBSET_2026_10_18.json'
        args=['experiments.bora_extension.acquire_tissue_catalog','--root',data,'--catalog',str(REPO/'experiments/bora_extension'/catalog),'--receipt-name',a.action+'-acquisition.json']
    elif a.action=='inventory':
        args=['experiments.bora_extension.bootstrap_tissue_sources','--data',data]
    elif a.action=='orca-download':
        args=['experiments.bora_extension.acquire_orca','--root',data,'--list',str(a.data.resolve()/'metadata/orca-provider-validation-list-v1.json')]
    elif a.action=='prepare':
        if a.dataset is None or a.out is None:p.error('prepare requires --dataset and --out')
        module='prepare_'+a.dataset if a.dataset in ('puma','histoseg','gleasonxai') else 'prepare_tissue_catalog'
        args=['experiments.bora_extension.'+module,'--data',data,'--output',str(a.out.resolve())]
        if module=='prepare_tissue_catalog':args+=['--dataset',a.dataset]
    else:
        if a.out is None:p.error('viewer requires --out')
        args=['experiments.bora_extension.tissue_acquisition_viewer','--data',data,'--out',str(a.out.resolve())]
    return subprocess.call([sys.executable,'-m']+args,cwd=REPO)


if __name__=='__main__':raise SystemExit(main())
