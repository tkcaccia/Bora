"""Small coarse-only, per-image neural segmenter; no pretrained downloads."""
import numpy as np
from .registry import PARAMETERS as P


def student(c):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(2);torch.manual_seed(P['random_seed']);rng=np.random.default_rng(P['random_seed'])
    channels=P['cnn_channels'];d=P['cnn_dilations']
    model=nn.Sequential(nn.Conv2d(6,channels,3,padding=d[0],dilation=d[0]),nn.ReLU(),
        nn.Conv2d(channels,channels,3,padding=d[1],dilation=d[1]),nn.ReLU(),
        nn.Conv2d(channels,channels,3,padding=d[2],dilation=d[2]),nn.ReLU(),nn.Conv2d(channels,c.k,1))
    optimizer=torch.optim.Adam(model.parameters(),lr=P['cnn_learning_rate'])
    image=np.concatenate([c.rgb,c.he],-1).astype(np.float32)/127.5-1
    patch=P['cnn_patch_px'];height,width=c.shape;pad_y=max(0,patch-height);pad_x=max(0,patch-width)
    image=np.pad(image,((0,pad_y),(0,pad_x),(0,0)),mode='reflect');truth=np.pad(c.core,((0,pad_y),(0,pad_x)),constant_values=-1)
    coordinates=[np.argwhere(c.core==i) for i in range(c.k)];history=[]
    for step in range(P['cnn_steps']):
        cls=step%c.k;points=coordinates[cls];y,x=points[rng.integers(len(points))]
        y0=int(np.clip(y-patch//2,0,image.shape[0]-patch));x0=int(np.clip(x-patch//2,0,image.shape[1]-patch))
        a=image[y0:y0+patch,x0:x0+patch];b=truth[y0:y0+patch,x0:x0+patch]
        rotation=int(rng.integers(4));a=np.rot90(a,rotation);b=np.rot90(b,rotation)
        if rng.random()<.5:a=a[:,::-1];b=b[:,::-1]
        a=a+rng.normal(0,.015,(1,1,6)).astype(np.float32)
        tensor=torch.tensor(a.transpose(2,0,1)[None].copy());target=torch.tensor(b[None].copy(),dtype=torch.long)
        optimizer.zero_grad();logits=model(tensor);loss=F.cross_entropy(logits,target,ignore_index=-1);loss.backward();optimizer.step()
        if step%20==0 or step==P['cnn_steps']-1:history.append(dict(step=step,cross_entropy=float(loss.detach())))
    model.eval()
    with torch.no_grad():
        values=torch.tensor(image[:height,:width].transpose(2,0,1)[None].copy())
        result=model(values).argmax(1)[0].numpy().astype(np.int32)
    c.diagnostics['B20']=dict(history=history,steps=P['cnn_steps'],trainable_parameters=sum(p.numel() for p in model.parameters()),
        training='original coarse interiors only; per-image fit, no fine labels, no cross-case pretrained model',receptive_field_um=(1+2*sum(d))*c.um)
    return result
