"""Portable entry point for the frozen colorectal whole-slide algorithm."""
from __future__ import annotations
import copy, json, math, os, time
from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image
from rasterio.windows import Window
from shapely import make_valid
from shapely.geometry import shape, mapping, box
from shapely.affinity import scale
from shapely.ops import unary_union
from . import selected_wsi as numerical
from .selected_wsi import sha, atomic, PROFILE
from .compact_geojson import geometry_agreement
from .mask_cleanup import remove_small_vector_parts, fill_small_vector_holes

FORMAT = "bora-wsi-colorectal-v1"
RESPONSE_FORMAT = "bora-slide-global-responses-v1"
MAX_VECTOR_REPAIR_PASSES = 8


def bound_file(base, value, name):
    """Resolve relative paths and bind their bytes before computation."""
    ref = {"path": value} if isinstance(value, str) else value
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError(f"{name} requires a file path")
    path = (base / ref["path"]).resolve()
    actual = sha(path)
    if ref.get("sha256", actual) != actual:
        raise ValueError(f"{name} checksum differs")
    return path, actual


def spacing(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError(f"{name} must contain two positive finite pixel spacings")
    return result


def read_case(path):
    import pyvips
    path = Path(path).resolve(); spec = json.loads(path.read_text())
    allowed = {"format", "source_frame_id", "image", "shape_yx", "group_grid",
        "probabilities", "response_receipt", "working_mpp_xy", "native_mpp_xy",
        "native_shape_yx", "analysis_native_bounds_xyxy", "quality_control"}
    if spec.get("format") != FORMAT or set(spec) - allowed:
        raise ValueError("Use a bora-wsi-colorectal-v1 manifest containing predictors only")
    frame = spec.get("source_frame_id")
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError("A complete analysis-frame identity is required")
    paths = {}; hashes = {}
    for key in ("image", "group_grid", "probabilities", "response_receipt"):
        paths[key], hashes[key] = bound_file(path.parent, spec[key], key)
    mpp = spacing(spec["working_mpp_xy"], "Working MPP")
    native_mpp = spacing(spec["native_mpp_xy"], "Original WSI MPP")
    if np.any(np.abs(mpp - .5) > .01):
        raise ValueError("Supply the calibrated approximately 0.5 µm/pixel H&E image")
    dims = spec["shape_yx"]; native_shape = spec["native_shape_yx"]
    if any(not isinstance(s, list) or len(s) != 2 or any(type(x) is not int or x < 1 for x in s)
           for s in (dims, native_shape)):
        raise ValueError("Working and original WSI shapes must be positive integer YX pairs")
    bounds = spec.get("analysis_native_bounds_xyxy", [0, 0, native_shape[1], native_shape[0]])
    if (not isinstance(bounds, list) or len(bounds) != 4 or any(type(v) is not int for v in bounds)
            or not 0 <= bounds[0] < bounds[2] <= native_shape[1]
            or not 0 <= bounds[1] < bounds[3] <= native_shape[0]):
        raise ValueError("Analysis bounds must lie within the original WSI")
    physical = (np.array(bounds[2:]) - bounds[:2]) * native_mpp
    if np.any(np.abs(np.array(dims[::-1]) * mpp - physical) > np.maximum(mpp, native_mpp) + 1e-6):
        raise ValueError("Working and original image frames have different physical extents")
    image = pyvips.Image.new_from_file(str(paths["image"]), access="random")
    if image.format != "uchar" or image.bands != 3 or image.height < dims[0] or image.width < dims[1]:
        raise ValueError("The full calibrated frame must be covered by a uint8 RGB image")
    grid = np.load(paths["group_grid"], allow_pickle=False)
    if grid.ndim != 2 or grid.dtype.kind not in "iu" or np.any(grid < 0) or grid.max() > 255:
        raise ValueError("Use a categorical group grid with zero for pure-background tiles")
    classes = np.unique(grid[grid > 0]); k = len(classes)
    if not 2 <= k <= 255 or not np.array_equal(classes, np.arange(1, k + 1)):
        raise ValueError("Anonymous group IDs must be contiguous integers 1..K")
    if dims[0] > grid.shape[0] * 224 or dims[1] > grid.shape[1] * 224:
        raise ValueError("The 224-pixel tile grid does not cover the full analysis frame")
    prob = np.load(paths["probabilities"], mmap_mode="r", allow_pickle=False)
    if prob.dtype not in (np.dtype("float16"), np.dtype("float32")) or prob.shape != (*np.array(grid.shape) * 16, k):
        raise ValueError("Group responses must cover the full registered 14-pixel token lattice")
    for y in range(0, prob.shape[0], 128):
        p = np.asarray(prob[y:y+128], np.float32)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError("Group responses must be finite probabilities")
        total = p.sum(2)
        if np.any((total != 0) & (np.abs(total - 1) > .003)):
            raise ValueError("Group response channels are not normalized")
    response = json.loads(paths["response_receipt"].read_text())
    if (response.get("format") != RESPONSE_FORMAT or response.get("status") != "COMPLETE"
            or response.get("source_frame_id") != frame
            or response.get("source_image_sha256") != hashes["image"]
            or response.get("group_grid_sha256") != hashes["group_grid"]
            or response.get("probability_sha256") != hashes["probabilities"]
            or response.get("shape_yxc") != list(prob.shape)
            or response.get("class_ids") != list(range(1, k+1))
            or response.get("one_fit_used_across_slide") is not True
            or response.get("independently_fitted_windows") is not False
            or not isinstance(response.get("model_sha256"), str)
            or len(response["model_sha256"]) != 64):
        raise ValueError("Responses are not bound to one frozen slide-wide model and frame")
    identity = dict(profile=PROFILE, source_frame_id=frame, groups=k, shape_yx=dims,
        mpp_xy=mpp.tolist(), native_mpp_xy=native_mpp.tolist(), native_shape_yx=native_shape,
        analysis_bounds_native_xyxy=bounds, input_sha256=hashes,
        manifest_sha256=sha(path), expert_annotation_used=False,
        response_model_sha256=response["model_sha256"])
    if 'quality_control' in spec:
        if spec['quality_control'].get('format') == 'bora-background-only-support-v1':
            from .background_support import bind_support
        else:
            from .grandqc_support import bind_support
        identity['quality_control']=bind_support(spec['quality_control'],path.parent,hashes['image'],[image.height,image.width],mpp)
    return image, prob, grid, identity


def polygonal(geometry):
    """Discard non-area clipping remnants, preserving all positive contours."""
    geometry = make_valid(geometry)
    def parts(g):
        if g.geom_type == "Polygon": return [g] if g.area > 1e-8 else []
        if g.geom_type in ("MultiPolygon", "GeometryCollection"):
            return [p for child in g.geoms for p in parts(child)]
        return []
    return unary_union(parts(geometry))


def native_vectors(working, identity):
    """Transform the completed global coverage into original WSI coordinates."""
    mpp = np.array(identity["mpp_xy"]); nmpp = np.array(identity["native_mpp_xy"])
    factor = mpp / nmpp; origin = identity["analysis_bounds_native_xyxy"][:2]
    from shapely.affinity import translate
    native = copy.deepcopy(working); reports = {}
    for key in ("boundary_fitting", "small_polygon_part_cleanup", "small_vector_hole_cleanup", "compact_boundary_report"):
        if key in native: reports[key] = native.pop(key)
    native["source_working_grid_vector_processing"] = dict(coordinate_space="calibrated_pixel_edges", mpp_xy=mpp.tolist(), reports=reports)
    native.update(coordinate_space="original_WSI_native_pixel_edges", native_mpp_xy=nmpp.tolist(),
        native_shape_yx=identity["native_shape_yx"], working_mpp_xy=mpp.tolist(),
        working_pixel_to_native_scale_xy=factor.tolist(), native_origin_xy=origin,
        physical_area_rule=dict(minimum_component_and_enclosed_hole_um2=400,
            scope="entire slide before diagnostic cropping", strictly_less_than=True))
    for f in native["features"]:
        g = polygonal(translate(scale(shape(f["geometry"]), xfact=factor[0], yfact=factor[1], origin=(0,0)), xoff=origin[0], yoff=origin[1]))
        f["geometry"] = mapping(g); f["properties"].update(area_pixels=float(g.area), area_um2=float(g.area*np.prod(nmpp)))
    before = copy.deepcopy(native)
    native = remove_small_vector_parts(native, 400/np.prod(nmpp))
    native = fill_small_vector_holes(native, 400/np.prod(nmpp))
    agreement = geometry_agreement(before, native)
    if (not agreement["all_classes_preserved"] or not agreement["all_polygons_valid"]
            or agreement["minimum_class_iou"] < .95 or agreement["changed_tissue_area_fraction"] > .025
            or agreement["interclass_overlap_area_pixels"] > 1e-6):
        raise ValueError("Original-coordinate cleanup violates the frozen geometry guards")
    old = {f["properties"]["value"]: shape(f["geometry"]) for f in before["features"]}
    delta = sum(shape(f["geometry"]).symmetric_difference(old[f["properties"]["value"]]).area for f in native["features"]) * np.prod(nmpp)
    if delta > 2: raise ValueError("Unexpected coordinate-repair change above 2 µm²")
    for f in native["features"]:
        g = shape(f["geometry"])
        if not g.is_valid or any(p.area*np.prod(nmpp) < 400-1e-6 for p in ([g] if g.geom_type=="Polygon" else g.geoms)):
            raise ValueError("Native polygon does not satisfy the global 400 µm² rule")
    native["native_coordinate_final_cleanup"] = dict(agreement=agreement,
        minimum_part_and_hole_um2=400, scope="entire slide before cropping",
        changed_symmetric_area_um2=float(delta))
    return native


def run_wsi(manifest, output, *, minimum_free_bytes=1024**3, memory_budget_bytes=12*1024**3):
    """Run the selected global method once. Resource limits do not alter it."""
    import pyvips, shutil
    output = Path(output)
    if output.exists(): raise FileExistsError("Use a fresh full-slide output directory")
    image, prob, grid, identity = read_case(manifest)
    h,w = identity["shape_yx"]; mpp = tuple(identity["mpp_xy"])
    if type(memory_budget_bytes) is not int or memory_budget_bytes < 10*h*w + 256*1024**2:
        raise ValueError("Memory budget is too small for the global component audit")
    if minimum_free_bytes < 0 or shutil.disk_usage(output.parent).free < minimum_free_bytes + 12*h*w:
        raise ValueError("Insufficient free space for full-slide staging and reserve")
    cv2.setNumThreads(4); pyvips.cache_set_max_mem(64<<20); pyvips.cache_set_max(0)
    output.mkdir(); atomic(output/"INPUTS_FROZEN.json", identity); started=time.monotonic()
    try:
        if 'quality_control' in identity:
            if identity['quality_control']['mode'] == 'background-only':
                from .background_support import stream_support
            else:
                from .grandqc_support import stream_support
            support=stream_support(identity['quality_control'],(h,w),output)
        else:
            support=numerical.tissue_support(image,(h,w),mpp,output)
        mask=numerical.predict(image,prob,grid,support,output,mpp)
        cleaned,cleanup=numerical.clean_global(mask,mpp,output/"global-cleaned-mask.npy");del mask;numerical.release()
        atomic(output/"PREDICTION_FROZEN.json",dict(status="COMPLETE",inputs_sha256=sha(output/"INPUTS_FROZEN.json"),mask_sha256=sha(output/"global-cleaned-mask.npy"),cleanup=cleanup,expert_annotation_used=False))
        data,conversion=numerical.fit_vectors(cleaned,mpp,output);del cleaned;numerical.release()
        final=numerical.raster_global(data,(h,w),output/"global-rasterized-mask.npy");repairs=[]
        # A local raster tie can move by one pixel after a polygon patch. Keep
        # repairing the same frozen prediction until the vector and raster
        # agree, with a finite guard against oscillation.
        for i in range(MAX_VECTOR_REPAIR_PASSES):
            checked,report=numerical.clean_global(final,mpp,output/f"global-raster-clean-{i}.npy")
            data,n=numerical.repair_raster(data,final,checked);repairs.append(dict(patches=n,cleanup=report))
            if n==0:del checked;break
            data=remove_small_vector_parts(data,400/np.prod(mpp));data=fill_small_vector_holes(data,400/np.prod(mpp))
            del final,checked;numerical.release();final=numerical.raster_global(data,(h,w),output/f"global-raster-repaired-{i}.npy")
        else:raise ValueError(f"Global vector/raster cleanup failed after {MAX_VECTOR_REPAIR_PASSES} passes")
        audit=numerical.audit(data,final,mpp);atomic(output/"GLOBAL_AREA_AUDIT.json",audit)
        data["full_slide_input_identity"]=identity;data["minimum_component_and_hole_um2"]=400
        atomic(output/"BORA_FULL_SLIDE.working.geojson",data)
        native=native_vectors(data,identity);atomic(output/"BORA_FULL_SLIDE.wsi.geojson",native)
        tifffile.imwrite(output/"BORA_FULL_SLIDE.mask-0p5um.tif",final,tile=(512,512),compression="deflate",bigtiff=True,resolution=(10000/mpp[0],10000/mpp[1]),resolutionunit="CENTIMETER")
        names=("BORA_FULL_SLIDE.working.geojson","BORA_FULL_SLIDE.wsi.geojson","BORA_FULL_SLIDE.mask-0p5um.tif","INPUTS_FROZEN.json","PREDICTION_FROZEN.json","SEAM_AUDIT.json","GLOBAL_AREA_AUDIT.json")
        if 'quality_control' in identity:
            names=names+(('BACKGROUND_SUPPORT_USED.json',) if identity['quality_control']['mode']=='background-only' else ('GRANDQC_SUPPORT_USED.json',))
        result=dict(status="COMPLETE",profile=PROFILE,input_identity=identity,entire_slide_refinement=True,
            global_component_cleanup=True,diagnostic_fields_used_in_inference=False,
            global_geometry_audit=audit,native_cleanup=native["native_coordinate_final_cleanup"],
            raster_repair=repairs,conversion=conversion,elapsed_seconds=time.monotonic()-started,
            files_sha256={name:sha(output/name) for name in names})
        atomic(output/"FULL_SLIDE_COMPLETE.json",result);return result
    except Exception as exc:
        atomic(output/"FAILED.json",dict(status="FAILED",profile=PROFILE,error=str(exc)))
        raise


def crop_wsi(result_folder, rectangle, output):
    """Crop a completed global mask and vector. No solver or cleanup is called."""
    import rasterio
    folder=Path(result_folder);output=Path(output)
    if output.exists():raise FileExistsError(output)
    rec=json.loads((folder/"FULL_SLIDE_COMPLETE.json").read_text())
    if rec.get("status")!="COMPLETE" or not rec.get("entire_slide_refinement"):
        raise ValueError("A verified global result is required")
    if len(rectangle)!=4 or any(type(v) is not int for v in rectangle):raise ValueError("Use integer working-grid X,Y,width,height")
    x,y,w,h=rectangle;hh,ww=rec["input_identity"]["shape_yx"]
    if min(x,y)<0 or min(w,h)<1 or x+w>ww or y+h>hh:raise ValueError("Crop is outside the full analysis frame")
    for name in ("BORA_FULL_SLIDE.mask-0p5um.tif","BORA_FULL_SLIDE.wsi.geojson"):
        if sha(folder/name)!=rec["files_sha256"][name]:raise ValueError("Global output bytes changed")
    with rasterio.open(folder/"BORA_FULL_SLIDE.mask-0p5um.tif") as src:mask=src.read(1,window=Window(x,y,w,h))
    native=json.loads((folder/"BORA_FULL_SLIDE.wsi.geojson").read_text());identity=rec["input_identity"]
    factor=np.array(identity["mpp_xy"])/np.array(identity["native_mpp_xy"]);ox,oy=identity["analysis_bounds_native_xyxy"][:2]
    window=box(ox+x*factor[0],oy+y*factor[1],ox+(x+w)*factor[0],oy+(y+h)*factor[1])
    cropped=copy.deepcopy(native);cropped["features"]=[];cropped["crop_only"]=True
    for f in native["features"]:
        g=polygonal(shape(f["geometry"]).intersection(window))
        if g.is_empty:continue
        item=copy.deepcopy(f);item["geometry"]=mapping(g);item["properties"].update(area_pixels=float(g.area),area_um2=float(g.area*np.prod(identity["native_mpp_xy"])));cropped["features"].append(item)
    output.mkdir(parents=True);Image.fromarray(mask).save(output/"output-mask.png");atomic(output/"crop.wsi.geojson",cropped)
    result=dict(status="COMPLETE",profile=PROFILE,working_pixel_crop_xywh=rectangle,
        global_mask_sha256=rec["files_sha256"]["BORA_FULL_SLIDE.mask-0p5um.tif"],
        refinement_run_for_this_crop=False,cleanup_run_for_this_crop=False,
        output_mask_sha256=sha(output/"output-mask.png"))
    atomic(output/"CROP_RECEIPT.json",result);return result
