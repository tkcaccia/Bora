"""Harmonic propagation, expansion cuts, merging, and spectral partitions."""
import heapq
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve,eigsh
from scipy.sparse.csgraph import maximum_flow,breadth_first_order
from sklearn.cluster import KMeans
from scipy.special import softmax
from .registry import PARAMETERS as P


def harmonic(c):
    g=c.region_graph();adj=g['adj'];lap=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
    anchor=(g['anchor']>=0)*P['random_walker_anchor'];rhs=P['random_walker_screen']*softmax(-g['u'],axis=-1)
    rhs[np.arange(g['n']),np.maximum(g['anchor'],0)]+=anchor
    p=spsolve(lap+sparse.diags(anchor+P['random_walker_screen']),rhs)
    return p.argmax(-1)[g['regions']].astype(np.int32)


def cut_binary(d0,d1,pairs,weights):
    """x=0 is source-side. Integer capacities approximate continuous costs."""
    n=len(d0);shift=np.minimum(d0,d1);d0=d0-shift;d1=d1-shift;scale=10000
    a,b=pairs.T
    row=np.r_[a,b,np.full(n,n),np.arange(n)];col=np.r_[b,a,np.arange(n),np.full(n,n+1)]
    values=np.r_[weights,weights,d1,d0]
    cap=sparse.coo_matrix((np.maximum(0,np.rint(values*scale)).astype(np.int64),(row,col)),shape=(n+2,n+2)).tocsr()
    flow=maximum_flow(cap,n,n+1);residual=cap-flow.flow;residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
    x=np.ones(n,bool);x[reachable[reachable<n]]=False;return x


def expansion(unary,pairs,weights,initial,cycles=3):
    labels=initial.copy();n,k=unary.shape;a,b=pairs.T
    def energy(l):return float(unary[np.arange(n),l].sum()+np.sum(weights*(l[a]!=l[b])))
    energies=[energy(labels)]
    for _ in range(cycles):
        changed=False
        for alpha in range(k):
            d0=unary[np.arange(n),labels].copy();d1=unary[:,alpha].copy()
            e00=weights*(labels[a]!=labels[b]);e01=weights*(labels[a]!=alpha);e10=weights*(alpha!=labels[b])
            pair=.5*(e01+e10-e00)
            np.add.at(d1,a,e10-e00-pair);np.add.at(d1,b,e01-e00-pair)
            move=cut_binary(d0,d1,pairs,np.maximum(pair,0));candidate=labels.copy();candidate[move]=alpha
            e=energy(candidate)
            if e<energies[-1]-1e-7:labels=candidate;energies.append(e);changed=True
        if not changed:break
    return labels,energies


def graph_cut(c):
    g=c.region_graph();u=g['u'].astype(float).copy();known=g['anchor']>=0
    u[known]+=100;u[np.flatnonzero(known),g['anchor'][known]]-=100
    labels,history=expansion(u,g['pairs'],g['weights']*P['graph_cut_weight'],u.argmin(-1),P['graph_cut_cycles'])
    c.diagnostics['B07']=dict(energies=history,integer_capacity_scale=10000)
    return labels[g['regions']]


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


def spectral(c):
    g=c.region_graph();adj=g['adj'];degree=np.asarray(adj.sum(1)).ravel();inv=1/np.sqrt(np.maximum(degree,1e-8))
    lap=sparse.eye(g['n'])-sparse.diags(inv)@adj@sparse.diags(inv)
    k=min(8,g['n']-2);_,v=eigsh(lap,k=k,sigma=-.001,which='LM',v0=np.ones(g['n']))
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
    clusters=KMeans(n_clusters=min(max(4,2*c.k),g['n']),n_init=5,random_state=P['random_seed']).fit_predict(v)
    labels=np.empty(g['n'],np.int32)
    for cluster in np.unique(clusters):
        use=clusters==cluster;votes=(g['votes'][use]*g['areas'][use,None]).sum(0)
        label=int(votes.argmax()) if votes.sum()>0 else int(np.average(g['u'][use],axis=0,weights=g['areas'][use]).argmin())
        labels[use]=label
    return labels[g['regions']]
