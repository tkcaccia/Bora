from dataclasses import dataclass
import json
import os
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
    max_side: int = 0
    def predict(self, image, core, outer, box):
        gray = rgb_float(image).mean(-1)
        original_shape = core.shape
        original_outer = outer
        scale = min(1.0, self.max_side / max(original_shape)) if self.max_side > 0 else 1.0
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


def load_label_prompts(value):
    """Load ``{label_id: text prompt}`` from a JSON file or mapping."""
    if isinstance(value, dict):
        raw = value
    else:
        path = Path(value or "")
        if not path.is_file():
            raise BackendError("--label-map must name an existing JSON file for PathSegmentor")
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"Cannot read PathSegmentor label map {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BackendError("PathSegmentor label map must be a JSON object")
    prompts = {}
    for key, prompt in raw.items():
        try:
            label = int(key)
        except (TypeError, ValueError) as exc:
            raise BackendError(f"Invalid PathSegmentor label ID: {key!r}") from exc
        if label <= 0 or not isinstance(prompt, str) or not prompt.strip():
            raise BackendError(f"Label {key!r} must have a non-empty prompt and a positive ID")
        prompts[label] = prompt.strip()
    if not prompts:
        raise BackendError("PathSegmentor label map is empty")
    return prompts


class PathSegmentorBackend:
    """Adapter for the official text-prompted PathSegmentor implementation."""

    def __init__(self, checkpoint=None, device="cuda", repo_dir=None, label_map=None,
                 config=None, inference_fn=None):
        self.prompts = load_label_prompts(label_map)
        self.current_label = None
        self.device = device
        self.model = None
        self._inference_fn = inference_fn
        if inference_fn is not None:  # Explicit injection used by unit/integration tests.
            return
        checkpoint_path = Path(checkpoint or "").resolve()
        repo_path = Path(repo_dir or "").resolve()
        if not checkpoint_path.is_file():
            raise BackendError("--checkpoint must name a PathSegmentor model_state_dict.pt file")
        if not repo_path.is_dir():
            raise BackendError("--repo-dir must name the official PathSegmentor repository")
        config_path = Path(config).resolve() if config else repo_path / "configs/pathsegmentor_inference.yaml"
        if not config_path.is_file():
            raise BackendError(f"PathSegmentor inference config not found: {config_path}")
        sys.path.insert(0, str(repo_path.resolve()))
        previous_cwd = Path.cwd()
        try:
            os.chdir(repo_path)
            import torch
            from modeling.BaseModel import BaseModel
            from modeling import build_model
            from utilities.arguments import load_opt_from_config_files
            from utilities.constants import PATHSEG_CLASSES
            from modeling.language.loss import vl_similarity
        except Exception as exc:
            raise BackendError(
                "PathSegmentor dependencies are unavailable; install the official environment") from exc
        finally:
            os.chdir(previous_cwd)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA unavailable; PathSegmentor's official release requires a CUDA device")
        torch_device = torch.device(device)
        try:
            opt = load_opt_from_config_files([str(config_path.resolve())])
            opt.update({"CUDA": device.startswith("cuda"), "device": torch_device,
                        "world_size": 1, "local_size": 1, "rank": 0, "local_rank": 0,
                        "env_info": "Bora single-process inference"})
            os.chdir(repo_path)
            model = BaseModel(opt, build_model(opt)).from_pretrained(str(checkpoint_path.resolve()))
            self.model = model.eval().to(torch_device)
            with torch.inference_mode():
                self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                    PATHSEG_CLASSES + ["background"], is_eval=True)
        except Exception as exc:
            raise BackendError(f"Unable to initialize PathSegmentor: {exc}") from exc
        finally:
            os.chdir(previous_cwd)
        self.torch = torch
        self.vl_similarity = vl_similarity

    def set_label(self, label):
        label = int(label)
        if label not in self.prompts:
            raise BackendError(
                f"PathSegmentor label {label} is missing from --label-map; "
                f"available labels: {sorted(self.prompts)}")
        self.current_label = label

    @staticmethod
    def _prepare_image(image):
        from PIL import Image
        rgb = np.clip(rgb_float(image) * 255, 0, 255).astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        side = max(h, w)
        top, left = (side - h) // 2, (side - w) // 2
        padded = np.zeros((side, side, 3), np.uint8)
        padded[top:top+h, left:left+w] = rgb
        resized = np.asarray(Image.fromarray(padded).resize((1024, 1024), Image.Resampling.BICUBIC))
        return resized, (h, w, side, top, left)

    def _official_inference(self, image, prompt):
        torch = self.torch
        resized, geometry = self._prepare_image(image)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).to(self.device)
        text = [prompt.replace("unspecified", "multiple")]
        data = {"image": tensor, "text": text, "height": 1024, "width": 1024}
        switches = self.model.model.task_switch
        switches.update({"spatial": False, "visual": False, "grounding": True, "audio": False})
        with torch.inference_mode():
            results, image_size, extra = self.model.model.evaluate_demo([data])
            pred_masks = results["pred_masks"][0]
            visual = results["pred_captions"][0]
            textual = extra["grounding_class"]
            visual = visual / (visual.norm(dim=-1, keepdim=True) + 1e-7)
            textual = textual / (textual.norm(dim=-1, keepdim=True) + 1e-7)
            temperature = self.model.model.sem_seg_head.predictor.lang_encoder.logit_scale
            matched = self.vl_similarity(visual, textual, temperature=temperature).max(0)[1]
            selected = pred_masks[matched, :, :]
            probability = torch.nn.functional.interpolate(
                selected[None], image_size[-2:], mode="bilinear", align_corners=False
            )[0, 0, :1024, :1024].sigmoid().cpu().numpy()
        h, w, side, top, left = geometry
        square = transform.resize(probability, (side, side), order=1, preserve_range=True,
                                  anti_aliasing=False)
        return square[top:top+h, left:left+w].astype(np.float32)

    def predict(self, image, core, outer, box):
        if self.current_label is None:
            raise BackendError("PathSegmentor backend did not receive a label ID")
        prompt = self.prompts[self.current_label]
        score = np.asarray(
            self._inference_fn(image, prompt) if self._inference_fn else
            self._official_inference(image, prompt), dtype=np.float32)
        if score.shape != outer.shape:
            score = transform.resize(score, outer.shape, order=1, preserve_range=True,
                                     anti_aliasing=False).astype(np.float32)
        score = np.clip(score, 0, 1)
        return (score >= 0.5) & outer, score


def make_backend(name, checkpoint=None, device="cuda", repo_dir=None, watershed_max_side=0,
                 label_map=None, pathsegmentor_config=None):
    if name == "watershed":
        return WatershedBackend(max_side=int(watershed_max_side))
    if name == "medsam":
        if not checkpoint:
            raise BackendError("--checkpoint is required for MedSAM")
        return MedSAMBackend(checkpoint, device, repo_dir)
    if name in ("pathsegmentor", "pathsegmentator"):
        return PathSegmentorBackend(checkpoint, device, repo_dir, label_map, pathsegmentor_config)
    raise BackendError(name)
