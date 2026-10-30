"""Exact global 4/8-connectivity with bounded strip work and int64 areas.

Local strip labels are joined across every seam, including diagonal seams for
8-connectivity. Strips never define component or hole boundaries. Only the
returned label lattice uses int32; areas and coordinate sums are accumulated
with 64-bit integers. An anonymous temporary mmap is released with its array.
"""
from __future__ import annotations
from array import array
import os,tempfile
import cv2
import numpy as np


def connected_large(binary,connectivity=8,*,strip_rows=512):
    if connectivity not in (4,8) or strip_rows<1:raise ValueError('Invalid connectivity or strip height')
    if binary.ndim!=2:raise ValueError('Expected binary YX image')
    h,w=map(int,binary.shape)
    if min(h,w)<1:raise ValueError('Empty image')
    directory=os.environ.get('BORA_COMPONENT_SCRATCH_DIR',tempfile.gettempdir())
    temporary=tempfile.TemporaryFile(dir=directory)
    labels=np.memmap(temporary,dtype=np.int32,mode='w+',shape=(h,w))
    parent=array('I',[0]);rank=array('B',[0]);chunks=[];moments=[]
    bg=np.array([w,h,-1,-1,0],np.int64);bg_moment=np.zeros(2,np.uint64)
    def find(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    def join(a,b):
        a,b=find(a),find(b)
        if a==b:return
        if rank[a]<rank[b]:a,b=b,a
        parent[b]=a
        if rank[a]==rank[b]:rank[a]+=1
    previous=None
    try:
        for y in range(0,h,strip_rows):
            block=np.asarray(binary[y:min(y+strip_rows,h)],np.uint8)
            n,local,stats,centroids=cv2.connectedComponentsWithStats(block,connectivity=connectivity)
            stats=stats.astype(np.int64)
            # Recover exact local integer coordinate sums from OpenCV's ratios;
            # each bounded strip sum is below the exact-integer float64 limit.
            moment=np.zeros((n,2),np.uint64)
            present=stats[:,cv2.CC_STAT_AREA]>0
            moment[present]=np.rint(centroids[present]*stats[present,cv2.CC_STAT_AREA,None]).astype(np.uint64)
            moment[:,1]+=stats[:,cv2.CC_STAT_AREA].astype(np.uint64)*np.uint64(y)
            if stats[0,cv2.CC_STAT_AREA]>0:
                bg[:2]=np.minimum(bg[:2],stats[0,:2]+[0,y])
                bg[2:4]=np.maximum(bg[2:4],stats[0,:2]+[0,y]+stats[0,2:4]-1)
                bg[4]+=stats[0,4];bg_moment+=moment[0]
            base=len(parent)-1;count=n-1
            if base+count>=np.iinfo(np.int32).max:raise OverflowError('Too many component labels')
            if count:
                parent.extend(range(base+1,base+count+1));rank.extend([0]*count)
                fg=stats[1:].copy();fg[:,1]+=y;fg[:,2:4]+=fg[:,:2]-1
                chunks.append(fg);moments.append(moment[1:])
                local[local>0]+=base
                labels[y:y+len(block)]=local
                current=local[0]
                if previous is not None:
                    pairs=[(previous,current)]
                    if connectivity==8:pairs.extend(((previous[:-1],current[1:]),(previous[1:],current[:-1])))
                    for left,right in pairs:
                        valid=(left>0)&(right>0)
                        if valid.any():
                            packed=(left[valid].astype(np.uint64)<<np.uint64(32))|right[valid].astype(np.uint64)
                            for value in np.unique(packed):join(int(value>>np.uint64(32)),int(value&np.uint64(0xffffffff)))
                previous=local[-1].copy()
            else:
                # New temporary files read as zero; do not allocate zero stripes.
                previous=np.zeros(w,np.int32)
        total=len(parent)-1
        if total:
            roots=np.fromiter((find(i) for i in range(1,total+1)),dtype=np.uint32,count=total)
            unique,inverse=np.unique(roots,return_inverse=True)
            mapping=np.zeros(total+1,np.int32);mapping[1:]=inverse+1;nlabels=len(unique)+1
            records=np.concatenate(chunks,axis=0);sums=np.concatenate(moments,axis=0)
            chunks.clear();moments.clear()
            stats_out=np.zeros((nlabels,5),np.int64);stats_out[:,0]=w;stats_out[:,1]=h;stats_out[:,2:4]=-1
            for column in (0,1):np.minimum.at(stats_out[:,column],mapping[1:],records[:,column])
            for column in (2,3):np.maximum.at(stats_out[:,column],mapping[1:],records[:,column])
            np.add.at(stats_out[:,4],mapping[1:],records[:,4])
            sums_out=np.zeros((nlabels,2),np.uint64)
            for column in (0,1):np.add.at(sums_out[:,column],mapping[1:],sums[:,column])
            for y in range(0,h,strip_rows):labels[y:y+strip_rows]=mapping[labels[y:y+strip_rows]]
        else:
            nlabels=1;stats_out=np.zeros((1,5),np.int64);sums_out=np.zeros((1,2),np.uint64)
        stats_out[0]=bg;sums_out[0]=bg_moment
        nonempty=stats_out[:,4]>0
        stats_out[nonempty,2:4]=stats_out[nonempty,2:4]-stats_out[nonempty,:2]+1
        stats_out[~nonempty,:4]=[-1,-1,0,0]
        centers=np.full((nlabels,2),np.nan,np.float64)
        centers[nonempty]=sums_out[nonempty]/stats_out[nonempty,4,None]
        labels.flush()
        return nlabels,labels,stats_out,centers
    finally:
        temporary.close()
