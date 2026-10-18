"""Three distinct native H&E tissue-support hypotheses for cycle11.

All preserve every photometrically non-white pixel. Only original white
pixels can be removed. The inherited common support is not consulted.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi,sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay,QhullError
from scipy.special import betaincc

from .diagnostics import white_regions


def grid_geometry(shape,spacing,step):
    h,w=shape;mx,my=np.asarray(spacing,float)
    if min(h,w)<1 or not np.isfinite([mx,my,step]).all() or min(mx,my,step)<=0:
        raise ValueError('Finite positive physical grid required')
    gh,gw=max(1,round(h*my/step)),max(1,round(w*mx/step))
    return (gh,gw),np.array([w*mx/gw,h*my/gh])


def sample_centres(array,shape):
    y=np.minimum(((np.arange(shape[0])+.5)*array.shape[0]/shape[0]).astype(int),array.shape[0]-1)
    x=np.minimum(((np.arange(shape[1])+.5)*array.shape[1]/shape[1]).astype(int),array.shape[1]-1)
    return array[y[:,None],x[None,:]]


def convex_support(white):
    stain=~white;contours,_=cv2.findContours(stain.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    envelope=np.zeros(white.shape,np.uint8)
    if contours:
        hull=cv2.convexHull(np.concatenate(contours))
        cv2.fillConvexPoly(envelope,hull,1)
    return stain|(envelope>0),dict(model='native convex stained-pixel envelope ablation',
        stained_boundary_points=sum(len(c) for c in contours))


def alpha_support(white,spacing,alpha=64.,step=8.):
    shape,grid_spacing=grid_geometry(white.shape,spacing,step)
    occupancy=cv2.resize((~white).astype(np.float32),shape[::-1],interpolation=cv2.INTER_AREA)>0
    y,x=np.nonzero(occupancy);points=np.c_[(x+.5)*grid_spacing[0],(y+.5)*grid_spacing[1]]
    envelope=np.zeros(white.shape,np.uint8);kept=total=0;degenerate=False
    if len(points)>=3:
        try:
            triangles=Delaunay(points).simplices
            vertices=points[triangles]
            a=np.linalg.norm(vertices[:,1]-vertices[:,0],axis=1)
            b=np.linalg.norm(vertices[:,2]-vertices[:,1],axis=1)
            c=np.linalg.norm(vertices[:,0]-vertices[:,2],axis=1)
            u=vertices[:,1]-vertices[:,0];v=vertices[:,2]-vertices[:,0]
            cross=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
            radius=np.divide(a*b*c,2*cross,out=np.full(len(a),np.inf),where=cross>1e-12)
            selected=vertices[radius<=alpha];total=len(triangles);kept=len(selected)
            # Physical cell centres map to native pixel-centre coordinates.
            native=np.rint((selected/np.asarray(spacing)-.5)*16).astype(np.int32)
            # Rasterize triangles individually: fillPoly with many contours
            # uses an even/odd rule that can erase touching triangles.
            for triangle in native:cv2.fillConvexPoly(envelope,triangle,1,shift=4)
        except QhullError:degenerate=True
    else:degenerate=True
    return (~white)|(envelope>0),dict(model='native Delaunay alpha-complex envelope',
        alpha_radius_um=alpha,occupancy_step_um=grid_spacing.tolist(),occupancy_nodes=len(points),
        triangles=total,retained_triangles=kept,degenerate_stain_geometry=degenerate,
        nonwhite_preserved_without_hull=True)


def occupancy_support(white,spacing,step=4.,width=64.,minimum=.05):
    shape,grid_spacing=grid_geometry(white.shape,spacing,step)
    stain=sample_centres(~white,shape).astype(np.float64)
    window=tuple(max(1,round(width/v)) for v in grid_spacing)
    sums=cv2.boxFilter(stain,-1,window,normalize=False,borderType=cv2.BORDER_CONSTANT)
    count=cv2.boxFilter(np.ones(shape,np.float64),-1,window,normalize=False,borderType=cv2.BORDER_CONSTANT)
    posterior=betaincc(.5+sums,.5+np.maximum(count-sums,0),minimum)
    belief=cv2.resize(posterior,white.shape[::-1],interpolation=cv2.INTER_LINEAR)
    return (~white)|(belief>.5),dict(model='native Beta-binomial stain occupancy',
        occupancy_step_um=grid_spacing.tolist(),window_pixels_xy=list(window),
        window_um_xy=(np.array(window)*grid_spacing).tolist(),prior_alpha_beta=[.5,.5],
        minimum_stained_fraction=minimum,posterior_probability_threshold=.5,
        observations='one native pixel at each physical occupancy-cell centre',
        minimum_window_observations=int(count.min()),maximum_window_observations=int(count.max()))


def escape_times(white,spacing):
    """Solve -L t=1 on exterior-connected white; border t=0, stain reflects.

    Unreachable white components receive infinite escape time, not a false
    zero solution from an unconstrained singular Neumann system.
    """
    white=np.asarray(white,bool);h,w=white.shape;mx,my=np.asarray(spacing,float)
    seeds=np.zeros(white.shape,bool);seeds[[0,-1]]=white[[0,-1]];seeds[:,[0,-1]]=white[:,[0,-1]]
    reachable=ndi.binary_propagation(seeds,structure=ndi.generate_binary_structure(2,1),mask=white)
    free=reachable&~seeds;n=int(free.sum());index=np.full((h,w),-1,np.int32);index[free]=np.arange(n)
    times=np.full((h,w),np.inf);times[seeds]=0;diagonal=np.zeros(n);ii=[];jj=[];vv=[]
    yy,xx=np.nonzero(free)
    for dy,dx,weight in [(0,1,1/mx**2),(0,-1,1/mx**2),(1,0,1/my**2),(-1,0,1/my**2)]:
        ny,nx=yy+dy,xx+dx;inside=(ny>=0)&(ny<h)&(nx>=0)&(nx<w)
        rows=np.flatnonzero(inside);ny,nx=ny[inside],nx[inside]
        connected=white[ny,nx];rows,ny,nx=rows[connected],ny[connected],nx[connected]
        diagonal[rows]+=weight
        columns=index[ny,nx];valid=columns>=0
        ii.extend(rows[valid]);jj.extend(columns[valid]);vv.extend(np.full(valid.sum(),-weight))
    residual=0.
    if n:
        matrix=sparse.coo_matrix((vv,(ii,jj)),shape=(n,n)).tocsc()+sparse.diags(diagonal,format='csc')
        solution=spsolve(matrix,np.ones(n))
        residual=float(np.max(np.abs(matrix@solution-1)))
        if not np.isfinite(solution).all() or solution.min()<-1e-8:
            raise ValueError('Invalid mean escape-time solution')
        times[free]=np.maximum(solution,0)
    return times,reachable,dict(free_white_nodes=n,border_white_seeds=int(seeds.sum()),
        unreachable_white_nodes=int((white&~reachable).sum()),poisson_max_residual=residual)


def escape_support(white,spacing,step=4.,threshold=32.**2):
    shape,grid_spacing=grid_geometry(white.shape,spacing,step)
    grid=sample_centres(white,shape)
    times,reachable,report=escape_times(grid,grid_spacing)
    finite=np.isfinite(times)
    numerator=cv2.resize(np.where(finite,times,0),white.shape[::-1],interpolation=cv2.INTER_LINEAR)
    weight=cv2.resize(finite.astype(float),white.shape[::-1],interpolation=cv2.INTER_LINEAR)
    interpolated=np.divide(numerator,weight,out=np.full(white.shape,np.inf),where=weight>1e-12)
    native_reachable=cv2.resize(reachable.astype(np.uint8),white.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    excluded=white&native_reachable&(interpolated<=threshold)
    return ~excluded,dict(report,model='native Poisson mean exterior escape time',
        occupancy_step_um=grid_spacing.tolist(),maximum_removed_escape_time_um2=threshold,
        time_coordinate='diffusion coordinate squared, not measured temporal seconds',
        white_graph='four-neighbour; border absorption and reflecting stain boundary')


def infer_support(key,he,spacing):
    white,_,policy=white_regions(he)
    if key=='P10':support,report=alpha_support(white,spacing)
    elif key=='P11':support,report=occupancy_support(white,spacing)
    elif key=='P12':support,report=escape_support(white,spacing)
    elif key=='B23':support,report=convex_support(white)
    else:raise ValueError('Unknown native-support hypothesis')
    if support.shape!=white.shape or np.any((~white)&~support):
        raise ValueError('A native-support hypothesis removed non-white tissue')
    return support,dict(report,white_policy=policy,own_native_support=True,
        inherited_common_clip_applied=False,reference_used=False,
        native_mpp_xy=list(map(float,spacing)),native_shape_yx=list(white.shape),
        retained_pixels=int(support.sum()),removed_white_pixels=int((~support).sum()))
