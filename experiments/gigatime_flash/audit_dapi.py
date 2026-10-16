#!/usr/bin/env python3
"""Audit Flash inference against the authors' paired test-patch DAPI references.

Diagnostic architecture alternatives are explicit hypotheses, not silent model
changes. The first five lexicographically sorted patches are calibration only;
the remaining patches can be used for confirmation after fixing a hypothesis.
"""
from __future__ import annotations

import argparse
import gzip
import importlib
import json
import pickle
from pathlib import Path
import sys
import types

import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.gigatime_flash.run_pilot import load_model


class NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        multi = importlib.import_module("numpy._core.multiarray")
        for prefix in ["numpy.core.multiarray", "numpy._core.multiarray"]:
            allowed[(prefix, "_reconstruct")] = multi._reconstruct
            allowed[(prefix, "scalar")] = multi.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Non-array pickle global blocked: {module}.{name}")
        return allowed[(module, name)]


def reference(path):
    with gzip.open(path, "rb") as f:
        data = NumpyOnlyUnpickler(f).load()
    shape = tuple(data["original_shape"])
    if shape != (556, 556, 23):
        raise ValueError(f"Unexpected binary label dimensions: {shape}")
    mask = np.unpackbits(data["comet_array_binary"], axis=-1)[..., :23].reshape(shape)
    nuclei = np.asarray(data["labels_dapi"]) > 0
    return mask, nuclei


def forward_hypothesis(self, x, *, cls_token=False, norm=False):
    outs = []
    x = self.encoder.patch_embed(x)
    grid = int(x.shape[1]**.5)
    pos = self._resize_pos_embed(self.encoder.pos_embed, (grid, grid))
    if cls_token:
        x = torch.cat((self.encoder.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + pos
    else:
        x = x + pos[:, 1:]
    x = self.encoder.norm_pre(self.encoder.patch_drop(x))
    for idx, block in enumerate(self.encoder.blocks):
        x = block(x)
        if idx in {3, 5, 8, 11}:
            out = self.encoder.norm(x) if norm else x
            outs.append(out[:, 1:] if cls_token else out)
    x = self.decoder4(self._tokens_to_patch(outs[3]))
    x = self.decoder3(x+self.skip3(self._tokens_to_patch(outs[2])))
    x = self.decoder2(x+self.skip2(self._tokens_to_patch(outs[1])))
    x = self.decoder1(x+self.skip1(self._tokens_to_patch(outs[0])))
    return self.final_conv(x)


def predict(rgb, model, normalization="imagenet", size=512, stride=256):
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    values = rgb.astype(np.float32)/255
    if normalization == "imagenet":
        values = (values-[.485, .456, .406])/[.229, .224, .225]
    elif normalization == "bgr_imagenet":
        values = (values[..., ::-1]-[.485, .456, .406])/[.229, .224, .225]
    elif normalization == "minus_one_one":
        values = values*2-1
    elif normalization != "zero_one":
        raise ValueError(normalization)
    coords = [(y, x) for y in range(0, size-255, stride) for x in range(0, size-255, stride)]
    accum, mass = np.zeros((23, size, size), np.float32), np.zeros((size, size), np.float32)
    with torch.inference_mode():
        for k in range(0, len(coords), 4):
            batch = coords[k:k+4]
            x = np.stack([values[y:y+256, x:x+256].transpose(2, 0, 1) for y, x in batch]).astype(np.float32)
            probs = model(torch.from_numpy(x).cuda()).sigmoid().cpu().numpy()
            for (y, x), patch in zip(batch, probs):
                accum[:, y:y+256, x:x+256] += patch
                mass[y:y+256, x:x+256] += 1
    return accum / mass


def metrics(pred, ref):
    truth = cv2.resize(ref.astype(np.uint8), pred.shape[::-1], interpolation=cv2.INTER_NEAREST)>0
    def corr(x, y):
        return float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) if x.std() and y.std() else None
    area = pred.shape[0]//8
    p8 = cv2.resize(pred, (area, area), interpolation=cv2.INTER_AREA)
    t8 = cv2.resize(truth.astype(np.float32), (area, area), interpolation=cv2.INTER_AREA)
    binary = pred > .5
    return {"auc": float(roc_auc_score(truth.ravel(), pred.ravel())),
            "average_precision": float(average_precision_score(truth.ravel(), pred.ravel())),
            "pearson_pixel": corr(pred, truth), "pearson_8px": corr(p8, t8),
            "dice_05": float(2*np.count_nonzero(binary & truth)/(binary.sum()+truth.sum())),
            "pred_mean_in_nuclei": float(pred[truth].mean()),
            "pred_mean_outside_nuclei": float(pred[~truth].mean())}


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model, info = load_model(args.weights)
    model.cuda().eval()
    paths = sorted(args.data.glob("*_he.png"))[:args.limit]
    hypotheses = [
        ("released_notebook", False, False, "imagenet", True),
        ("with_cls", True, False, "imagenet", True),
        ("normalized_intermediates", False, True, "imagenet", True),
        ("cls_and_normalized", True, True, "imagenet", True),
        ("official_fallback_no_patch_lora", False, False, "imagenet", False),
        ("rgb_zero_one", False, False, "zero_one", True),
        ("rgb_minus_one_one", False, False, "minus_one_one", True),
        ("bgr_diagnostic", False, False, "bgr_imagenet", True),
    ]
    records = []
    for name, use_cls, norm, normalization, patch_lora in hypotheses:
        model.forward = types.MethodType(lambda self, x, c=use_cls, n=norm:
                                        forward_hypothesis(self, x, cls_token=c, norm=n), model)
        model.encoder.patch_embed.proj.scaling = 2 if patch_lora else 0
        for p in paths:
            key = p.name.replace("_he.png", "")
            rgb = np.asarray(Image.open(p).convert("RGB"))
            ref, nuclei = reference(args.data/f"{key}_comet_binary_thres_labels.pkl.gz")
            maps = predict(rgb, model, normalization)
            row = {"hypothesis": name, "sample": key,
                   "DAPI_vs_released_binary": metrics(maps[0], ref[..., 0]),
                   "DAPI_vs_reference_nuclei": metrics(maps[0], nuclei)}
            records.append(row)
            np.savez_compressed(args.output/f"{key}.{name}.npz", dapi=maps[0], ck=maps[16])
        subset = [r["DAPI_vs_reference_nuclei"] for r in records if r["hypothesis"] == name]
        print(json.dumps({"hypothesis": name, "mean": {k: float(np.mean([r[k] for r in subset]))
                                         for k in subset[0]}}), flush=True)
    (args.output/"audit.json").write_text(json.dumps({"model": info,
        "calibration_only": True, "samples": [p.name for p in paths], "results": records}, indent=2))


if __name__ == "__main__":
    main()
