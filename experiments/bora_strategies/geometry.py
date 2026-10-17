"""Topology-constrained edits and a shared smooth mask deformation."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from .registry import PARAMETERS as P

OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def simple_table():
    table=np.zeros(256,bool)
    for code in range(256):
        a=np.zeros((3,3),bool)
        for i,(dy,dx) in enumerate(OFFSETS):a[dy+1,dx+1]=bool(code&(1<<i))
        _,fg=ndi.label(a,np.ones((3,3)))
        background=~a;background[1,1]=False
        components,_=ndi.label(background,ndi.generate_binary_structure(2,1))
        near=np.unique(components[[0,1,1,2],[1,0,2,1]]);bg=np.count_nonzero(near)
        table[code]=fg==1 and bg==1
    return table


SIMPLE=simple_table()


def topology(c):
    answer=c.initial.copy();target=c.u.argmin(-1);yy,xx=np.indices(c.shape)
    eligible=c.anchors<0;eligible[[0,-1]]=False;eligible[:,[0,-1]]=False;changes=[]
    for _ in range(P['topology_iterations']):
        changed=0
        for py in range(3):
            for px in range(3):
                codes=np.zeros((*c.shape,c.k),np.uint16);padded=np.pad(answer,1,mode='edge')
                for bit,(dy,dx) in enumerate(OFFSETS):
                    values=padded[1+dy:1+dy+c.shape[0],1+dx:1+dx+c.shape[1]]
                    for i in range(c.k):codes[...,i]|=(values==i).astype(np.uint16)<<bit
                old=np.take_along_axis(codes,answer[...,None],axis=-1)[...,0]
                new=np.take_along_axis(codes,target[...,None],axis=-1)[...,0]
                allowed=eligible&(answer!=target)&(yy%3==py)&(xx%3==px)&SIMPLE[old]&SIMPLE[new]
                answer[allowed]=target[allowed];changed+=int(allowed.sum())
        changes.append(changed)
        if not changed:break
    c.diagnostics['B14']=dict(simple_point_changes_by_sweep=changes,connectivity='8-connected classes / 4-connected complement',image_border_fixed=True)
    return answer


def elastic(c):
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(P['random_seed'])
    factor=4;low=(max(8,round(c.shape[0]/factor)),max(8,round(c.shape[1]/factor)))
    initial=np.eye(c.k,dtype=np.float32)[c.initial]
    soft=cv2.GaussianBlur(initial,(0,0),2/c.um)
    if soft.ndim==2:soft=soft[...,None]
    source=torch.tensor(cv2.resize(soft,tuple(low[::-1]),interpolation=cv2.INTER_AREA).reshape(*low,c.k).transpose(2,0,1)[None])
    cost=torch.tensor(cv2.resize(c.u,tuple(low[::-1]),interpolation=cv2.INTER_AREA).reshape(*low,c.k).transpose(2,0,1)[None])
    gh=max(3,round(c.shape[0]*c.mpp_xy[1]/P['elastic_control_spacing_um']));gw=max(3,round(c.shape[1]*c.mpp_xy[0]/P['elastic_control_spacing_um']))
    control=torch.nn.Parameter(torch.zeros((1,2,gh,gw)));optimizer=torch.optim.Adam([control],lr=.07)
    yy,xx=torch.meshgrid(torch.linspace(-1,1,low[0]),torch.linspace(-1,1,low[1]),indexing='ij');grid=torch.stack([xx,yy],-1)[None]
    limits=torch.tensor([2*P['elastic_max_displacement_um']/(c.shape[1]*c.mpp_xy[0]),2*P['elastic_max_displacement_um']/(c.shape[0]*c.mpp_xy[1])],dtype=torch.float32)[None,:,None,None]
    history=[]
    for step in range(P['elastic_iterations']):
        optimizer.zero_grad();bounded=torch.tanh(control);flow=F.interpolate(bounded*limits,size=low,mode='bilinear',align_corners=True)
        warped=F.grid_sample(source,grid+flow.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=True)
        appearance=(warped*cost).sum(1).mean()
        bending=(bounded[:,:,1:]-bounded[:,:,:-1]).square().mean()+(bounded[:,:,:,1:]-bounded[:,:,:,:-1]).square().mean()
        loss=appearance+.12*bending+.005*bounded.square().mean();loss.backward();optimizer.step()
        if step%20==0 or step==P['elastic_iterations']-1:history.append(dict(step=step,loss=float(loss.detach()),appearance=float(appearance.detach())))
    with torch.no_grad():
        fy,fx=torch.meshgrid(torch.linspace(-1,1,c.shape[0]),torch.linspace(-1,1,c.shape[1]),indexing='ij');fine_grid=torch.stack([fx,fy],-1)[None]
        flow=F.interpolate(torch.tanh(control)*limits,size=c.shape,mode='bilinear',align_corners=True)
        onehot=torch.tensor(initial.transpose(2,0,1)[None]);warped=F.grid_sample(onehot,fine_grid+flow.permute(0,2,3,1),mode='bilinear',padding_mode='border',align_corners=True)
        result=warped.argmax(1)[0].numpy().astype(np.int32)
    c.diagnostics['B18']=dict(history=history,control_shape_yx=[gh,gw],maximum_displacement_um=P['elastic_max_displacement_um'],invertibility_guaranteed=False)
    return result
