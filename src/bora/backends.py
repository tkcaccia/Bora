from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from skimage import filters, segmentation, transform


class BackendError(RuntimeError):
    pass


def rgb_float(image):
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / np.iinfo(arr.dtype).max
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, (1, 99))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


@dataclass
class WatershedBackend:
    compactness: float = 0.001
    max_side: int = 256
    def predict(self, image, core, outer, box):
        gray = rgb_float(image).mean(-1)
        original_shape = core.shape
        original_outer = outer
        scale = min(1.0, self.max_side / max(original_shape))
        if scale < 1:
            shape = tuple(max(1, round(v*scale)) for v in original_shape)
            gray = transform.resize(gray, shape, preserve_range=True, anti_aliasing=True)
            core = transform.resize(core, shape, order=0, preserve_range=True).astype(bool)
            outer = transform.resize(outer, shape, order=0, preserve_range=True).astype(bool)
        elevation = filters.sobel(gray)
        markers = np.zeros(core.shape, np.int8)
        markers[~outer], markers[core] = 1, 2
        pred = segmentation.watershed(elevation, markers, compactness=self.compactness) == 2
        confidence = 1 - elevation / max(float(elevation.max(initial=0)), 1e-6)
        if pred.shape != original_shape:
            pred = transform.resize(pred, original_shape, order=0, preserve_range=True).astype(bool)
            confidence = transform.resize(confidence, original_shape, order=1, preserve_range=True)
        return pred & original_outer, confidence.astype(np.float32)


class MedSAMBackend:
    def __init__(self, checkpoint, device="cuda", repo_dir=None):
        if not Path(checkpoint).is_file():
            raise BackendError(f"Checkpoint not found: {checkpoint}")
        if repo_dir:
            sys.path.insert(0, repo_dir)
        try:
            import torch
            from segment_anything import sam_model_registry
        except Exception as exc:
            raise BackendError("Install the MedSAM dependencies first") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA unavailable; use --device cpu")
        self.torch, self.device = torch, device
        self.model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device).eval()

    def predict(self, image, core, outer, box):
        torch, h, w = self.torch, *image.shape[:2]
        rgb = transform.resize(rgb_float(image), (1024, 1024), preserve_range=True).astype(np.float32)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        scale = np.array([1024/w, 1024/h, 1024/w, 1024/h], np.float32)
        prompt = torch.as_tensor(np.array(box) * scale, device=self.device).float()[None, None]
        with torch.inference_mode():
            emb = self.model.image_encoder(tensor)
            sparse, dense = self.model.prompt_encoder(points=None, boxes=prompt, masks=None)
            logits, _ = self.model.mask_decoder(image_embeddings=emb, image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            prob = torch.nn.functional.interpolate(torch.sigmoid(logits), (h, w), mode="bilinear", align_corners=False)[0, 0]
        score = prob.cpu().numpy().astype(np.float32)
        return (score >= .5) & outer, score


def make_backend(name, checkpoint=None, device="cuda", repo_dir=None):
    if name == "watershed":
        return WatershedBackend()
    if name == "medsam":
        if not checkpoint:
            raise BackendError("--checkpoint is required for MedSAM")
        return MedSAMBackend(checkpoint, device, repo_dir)
    if name == "pathsegmentor":
        raise BackendError("PathSegmentor needs a text prompt/class map; use medsam or watershed in this release")
    raise BackendError(name)
