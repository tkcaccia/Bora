"""Retained hierarchical image-region merging."""
import heapq
import numpy as np
from .registry import PARAMETERS as P










def merge(c):
    g=c.region_graph();n=g['n'];parent=np.arange(n);means=g['colour'].copy();area=g['areas'].astype(float);anchor=g['anchor'].copy()
    neighbors=[set(g['adj'].indices[g['adj'].indptr[i]:g['adj'].indptr[i+1]]) for i in range(n)]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def distance(a,b):return float(np.linalg.norm(means[a]-means[b]))
    heap=[(distance(a,b),int(a),int(b)) for a,b in g['pairs']];heapq.heapify(heap);merges=0
    while heap:
        dist,a,b=heapq.heappop(heap);a=root(a);b=root(b)
        if a==b or (anchor[a]>=0 and anchor[b]>=0 and anchor[a]!=anchor[b]):continue
        current=distance(a,b)
        if abs(current-dist)>1e-5:heapq.heappush(heap,(current,a,b));continue
        if current>P['merge_LAB_threshold']:continue
        if area[a]<area[b]:a,b=b,a
        means[a]=(area[a]*means[a]+area[b]*means[b])/(area[a]+area[b]);area[a]+=area[b];parent[b]=a
        if anchor[a]<0:anchor[a]=anchor[b]
        neighbors[a]|=neighbors[b];neighbors[a]={root(z) for z in neighbors[a]}-{a}
        for z in neighbors[a]:heapq.heappush(heap,(distance(a,z),a,z))
        merges+=1
    roots=np.array([root(i) for i in range(n)]);labels=np.empty(n,np.int32)
    for r in np.unique(roots):
        members=roots==r;labels[members]=anchor[r] if anchor[r]>=0 else np.average(g['u'][members],axis=0,weights=g['areas'][members]).argmin()
    c.diagnostics['B12']=dict(initial_regions=n,remaining_regions=int(len(np.unique(roots))),merges=merges)
    return labels[g['regions']]
