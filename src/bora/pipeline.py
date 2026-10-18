"""The single supported pipeline: RGB9+19 -> Bora-AF -> Neighbor context -> compact vectors.

Virchow2 inference is upstream. This CPU package consumes calibrated spatial
tokens; it contains no GPU launcher, scheduler, retry loop or automatic jobs.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
import tifffile
from rasterio.features import rasterize
from shapely.geometry import shape
from threadpoolctl import threadpool_limits
from .appearance import refine_appearance
from .classifier import NeighborContextClassifier
from .conversion import convert_rgb, observations
from .groups import describe_groups, spacing_xy
from .multicpu_geojson import convert
from .compact_geojson import compact
from .mask_cleanup import fill_small_regions, finalize_vector_regions

ENCODER_REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
ENCODER_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_file(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _image(path):
    if Path(path).suffix.lower() in (".tif", ".tiff", ".btf"):
        data = tifffile.imread(path)
    else:
        with Image.open(path) as im:
            data = np.asarray(im.convert("RGB"))
    if data.ndim != 3 or data.shape[-1] != 3 or data.dtype != np.uint8:
        raise ValueError("Expected a two-dimensional uint8 RGB H&E image")
    return data


@dataclass
class Case:
    image: np.ndarray
    coarse: np.ndarray
    tissue: np.ndarray
    tokens: np.ndarray
    mpp_xy: np.ndarray
    native_mpp_xy: np.ndarray
    native_tissue: np.ndarray
    rgb: np.ndarray | None = None
    palette: np.ndarray | None = None
    variants: dict | None = None
    provenance: dict | None = None

    def validate(self):
        image, coarse = np.asarray(self.image), np.asarray(self.coarse)
        self.mpp_xy = spacing_xy(self.mpp_xy)
        self.native_mpp_xy = spacing_xy(self.native_mpp_xy)
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or coarse.shape != image.shape[:2] or coarse.dtype.kind not in "iu"
                or np.any(coarse < 0) or np.any(coarse > 255) or not np.any(coarse > 0)
                or np.asarray(self.tissue).shape != coarse.shape):
            raise ValueError("Aligned uint8 H&E, integer coarse IDs in 0..255 and tissue support required")
        if min(coarse.shape) < 224 or np.any(np.abs(self.mpp_xy - .5) > .01):
            raise ValueError("Use a calibrated approximately 0.5 um/px working image, at least224 pixels per axis")
        tokens = np.asarray(self.tokens)
        expected = tuple((np.asarray(coarse.shape) + 13) // 14)
        if (tokens.shape != (*expected, 1280) or tokens.dtype.kind != "f"
                or not np.isfinite(tokens).all()):
            raise ValueError("Full finite 1280-channel Virchow2 tokens on the padded 14-pixel grid required")
        if np.asarray(self.native_tissue).ndim != 2:
            raise ValueError("Two-dimensional native tissue support required")
        native_extent = np.asarray(self.native_tissue.shape[::-1]) * self.native_mpp_xy
        working_extent = np.asarray(coarse.shape[::-1]) * self.mpp_xy
        if not np.allclose(native_extent, working_extent, rtol=1e-6, atol=1e-6):
            raise ValueError("Native and working pixel spacings must describe the same physical field")
        summaries, labels = observations(tokens, coarse)
        if len(summaries) < 16 or len(np.unique(labels)) < 2:
            raise ValueError("At least16 eligible coarse tiles and two coarse groups required")
        if not np.array_equal(np.unique(labels), np.unique(coarse[coarse > 0])):
            raise ValueError("Each coarse group must have a supported tile observation")
        if self.rgb is not None and (np.asarray(self.rgb).shape != image.shape or self.rgb.dtype != np.uint8):
            raise ValueError("Cached RGB9+19 must be aligned uint8 RGB")
        return self


def load_case(path):
    """Read only declared predictors. Fine annotation paths are not accepted."""
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    allowed = {"case", "image", "coarse", "tissue", "tokens", "mpp_xy", "native_mpp_xy",
               "native_tissue", "evidence_rgb", "palette", "variants", "encoder_revision", "encoder_sha256"}
    if set(spec) - allowed:
        raise ValueError("Unknown case fields: " + ", ".join(sorted(set(spec) - allowed)))
    if spec.get("encoder_revision") != ENCODER_REVISION or spec.get("encoder_sha256") != ENCODER_SHA256:
        raise ValueError("Declare the audited Virchow2 revision and checkpoint SHA256")
    hashes = {"case_manifest": digest(path)}
    def file(value, name):
        info = {"path": value} if isinstance(value, str) else value
        p = (path.parent / info["path"]).resolve()
        actual = digest(p)
        if info.get("sha256", actual) != actual:
            raise ValueError("Input checksum mismatch: " + name)
        hashes[name] = actual
        return p
    image = _image(file(spec["image"], "image"))
    coarse = tifffile.imread(file(spec["coarse"], "coarse"))
    tissue = tifffile.imread(file(spec["tissue"], "tissue")) > 0
    tokens = np.load(file(spec["tokens"], "tokens"), mmap_mode="r", allow_pickle=False)
    native = (tifffile.imread(file(spec["native_tissue"], "native_tissue")) > 0
              if "native_tissue" in spec else tissue.copy())
    rgb = _image(file(spec["evidence_rgb"], "evidence_rgb")) if "evidence_rgb" in spec else None
    variants = {}
    if rgb is None:
        for name in ("scale_1", "scale_2"):
            v = spec["variants"][name]
            variant_spacing = spacing_xy(v["mpp_xy"])
            extent = np.asarray(v["shape_yx"])[::-1] * variant_spacing
            if not np.allclose(extent, np.asarray(image.shape[:2][::-1]) * spacing_xy(spec["mpp_xy"]), rtol=1e-6):
                raise ValueError("Scale feature physical field differs from the base image")
            factor = 2 if name == "scale_1" else 4
            if not np.allclose(variant_spacing, factor * .5, rtol=.03):
                raise ValueError("Scale features must represent approximately1 and2 um/px")
            variants[name] = dict(values=np.load(file(v["features"], name), mmap_mode="r", allow_pickle=False),
                shape_yx=v["shape_yx"], selected_dimensions=v.get("selected_dimensions"))
    case = Case(image, coarse, tissue, tokens, spacing_xy(spec["mpp_xy"]),
        spacing_xy(spec.get("native_mpp_xy", spec["mpp_xy"])), native, rgb,
        np.asarray(spec["palette"]) if "palette" in spec else None, variants,
        dict(input_sha256=hashes, encoder_revision=ENCODER_REVISION, encoder_sha256=ENCODER_SHA256,
             fine_annotations_read=False, encoder_inference_included=False))
    return case.validate()


def prepare(case):
    """Refine geometry and build Neighbor context descriptors, before any semantic readout."""
    case.validate()
    start = perf_counter()
    if case.rgb is None:
        rgb, conversion = convert_rgb(case.tokens, case.coarse, case.palette, case.variants)
    else:
        rgb, conversion = case.rgb, dict(conversion="RGB9+19", supplied_cached_rgb=True,
                                        cached_formula_not_recomputed=True)
    rgb_seconds = perf_counter() - start
    start = perf_counter()
    groups, details = refine_appearance(rgb, case.coarse, case.tissue, case.mpp_xy, he=case.image,
        output_shape_yx=case.native_tissue.shape, output_tissue=case.native_tissue)
    refine_seconds = perf_counter() - start
    start = perf_counter()
    regions, ids, x, mass = describe_groups(case.tokens, groups, case.coarse.shape, case.native_mpp_xy)
    pool_seconds = perf_counter() - start
    return dict(regions=regions, group_ids=ids, descriptors=x, area_pixels=mass, rgb=rgb,
        details=dict(appearance=details, conversion=conversion,
            timing_seconds=dict(rgb_conversion=rgb_seconds, refinement=refine_seconds, pooling_and_context=pool_seconds)))


def _mask(path, data, mpp):
    tifffile.imwrite(path, data, ome=True, compression="deflate", photometric="minisblack",
        metadata={"axes": "YX", "PhysicalSizeX": float(mpp[0]), "PhysicalSizeY": float(mpp[1]),
                  "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})


def export_mask(mask, folder, mpp, names):
    start = perf_counter()
    cleaned, cleanup = fill_small_regions(mask, 20)
    temporary = folder / "export-input.tif"
    tifffile.imwrite(temporary, cleaned, compression="deflate")
    report = convert(temporary, folder / "mask.geojson", page=0, min_area=500., simplify=2.,
                     workers=1, striped_native=False)
    temporary.unlink()
    data = json.loads((folder / "mask.geojson").read_text())
    data.update(native_shape_yx=list(mask.shape), native_mpp_xy=list(map(float, mpp)),
                coordinate_space="native_image_pixel_edges", origin="top_left",
                minimum_component_area_pixels=500., simplification_tolerance_pixels=2.)
    for f in data["features"]:
        value = int(f["properties"]["value"])
        g = shape(f["geometry"])
        f["properties"].update(name=names[value], classification={"name": names[value]},
            predicted_tissue_id=value, predicted_tissue=names[value], objectType="annotation",
            area_pixels=float(g.area), area_um2=float(g.area * np.prod(mpp)),
            tissue_identity_source="Neighbor context, native fastPLS")
    data["small_region_cleanup"] = cleanup
    data = finalize_vector_regions(data, mask.shape, 20)
    data = compact(data, mpp, mask.shape)
    geometries = [shape(f["geometry"]) for f in data["features"]]
    overlap = sum(geometries[i].intersection(geometries[j]).area
                  for i in range(len(geometries)) for j in range(i))
    if overlap > 1e-7 or not all(g.is_valid for g in geometries):
        raise ValueError("Final vector coverage is invalid or overlaps")
    pairs = [(f["geometry"], int(f["properties"]["value"])) for f in data["features"]]
    final = (rasterize(pairs, out_shape=mask.shape, fill=0, dtype=np.uint8)
             if pairs else np.zeros(mask.shape, np.uint8))
    _mask(folder / "mask.ome.tif", final, mpp)
    json_file(folder / "mask.geojson", data)
    return dict(converter=report, export_seconds=perf_counter()-start,
                interclass_overlap_area_pixels=float(overlap), final_raster_matches_geojson=True)


def run(case, model, output):
    """Publish a complete atomic result; no ground-truth input or model selector."""
    if not isinstance(model, NeighborContextClassifier):
        raise TypeError("The supported classifier is NeighborContextClassifier")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    cv2.setNumThreads(2)
    with threadpool_limits(limits=2), tempfile.TemporaryDirectory(prefix=".bora-", dir=output.parent) as temp:
        folder = Path(temp) / "result"
        folder.mkdir()
        prepared = prepare(case)
        t = perf_counter()
        classes = model.predict(prepared["descriptors"])
        head_seconds = perf_counter() - t
        lookup = np.r_[np.uint8(0), classes]
        mask = lookup[prepared["regions"]]
        _mask(folder / "groups.ome.tif", prepared["regions"].astype(np.uint32), case.native_mpp_xy)
        _mask(folder / "classified-before-export.ome.tif", mask, case.native_mpp_xy)
        Image.fromarray(prepared["rgb"]).save(folder / "converted-rgb.png")
        report = export_mask(mask, folder, case.native_mpp_xy, model.names_)
        report.update(prepared["details"], classifier="Neighbor context", native_head_count=1,
            classifier_seconds=head_seconds, model_load_refit_seconds=model.fit_seconds_,
            pipeline_wall_seconds_excluding_model_load=perf_counter()-start,
            provenance=case.provenance or {}, fine_annotations_read=False,
            working_mpp_xy=case.mpp_xy.tolist(), native_mpp_xy=case.native_mpp_xy.tolist(),
            groups=[dict(anonymous_id=int(g), predicted_class=int(c), tissue=model.names_[int(c)],
                         native_area_pixels=int(a)) for g, c, a in zip(prepared["group_ids"], classes, prepared["area_pixels"])],
            output_sha256={p.name: digest(p) for p in folder.iterdir()})
        json_file(folder / "report.json", report)
        if output.exists():
            raise FileExistsError(output)
        os.rename(folder, output)
    return report
