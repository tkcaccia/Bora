import contextlib,importlib.util,sys,time,types
from pathlib import Path
import numpy as np

def test_raw_encoder_retains_every_channel_without_bora_labels(monkeypatch):
 import pyvips
 spec=importlib.util.spec_from_file_location('full_encoder',Path(__file__).parents[1]/'benchmark/encode_spatial_cache.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 class Tensor:
  def __init__(self,a):self.a=np.asarray(a)
  def to(self,*a,**k):return self
  def float(self):return self
  def cpu(self):return self
  def numpy(self):return self.a
  def __truediv__(self,v):return Tensor(self.a/(v.a if isinstance(v,Tensor) else v))
  def __sub__(self,v):return Tensor(self.a-(v.a if isinstance(v,Tensor) else v))
  def __getitem__(self,k):return Tensor(self.a[k])
  @property
  def shape(self):return self.a.shape
  @property
  def ndim(self):return self.a.ndim
 torch=types.SimpleNamespace(from_numpy=Tensor,inference_mode=contextlib.nullcontext,autocast=lambda *a,**k:contextlib.nullcontext(),float16=None)
 monkeypatch.setitem(sys.modules,'torch',torch)
 def model(t):return Tensor(np.broadcast_to(np.arange(1280,dtype=np.float32),(len(t.a),261,1280)))
 img=pyvips.Image.black(224,224,bands=3).cast('uchar')
 values,windows=m.encode_row(img,0,np.ones((1,1),bool),np.arange(1280),model,Tensor(0),Tensor(1),time.monotonic()+60)
 assert windows==1 and values.shape==(16,16,1280)
 assert np.all(values==np.arange(1280,dtype=np.float16))
