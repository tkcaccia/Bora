"""Frozen slide-wide RGB9+19 fitting and projection, without diagnostic refits."""
from __future__ import annotations
import json, os, colorsys
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .selected_wsi import sha, atomic, PROFILE
from .wsi_frozen import bound_file, RESPONSE_FORMAT
POSITIONS = (3, 7, 11, 14)


def tile_summaries(grid: np.memmap, factor: int, eligible: np.ndarray) -> np.ndarray:
    records = np.empty((int(eligible.sum()), 64), np.float32)
    extent = 16 // factor
    for i, (r, c) in enumerate(np.argwhere(eligible)):
        block = np.asarray(grid[r * extent:(r + 1) * extent,
                                c * extent:(c + 1) * extent], np.float32)
        if block.shape != (extent, extent, 64) or not np.isfinite(block).all():
            raise ValueError(f'Eligible tile {r},{c} lacks factor-{factor} spatial tokens')
        records[i] = block.mean(axis=(0, 1))
    return records

def fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(values)
    mean = scaler.mean_.astype(np.float32)
    scale = np.maximum(scaler.scale_, .05).astype(np.float32)
    return mean, scale, ((values - mean) / scale).astype(np.float32)

def persistence_positions(zs: dict[int, np.ndarray], labels: np.ndarray) -> np.ndarray:
    reference = zs[1]
    correlation = []
    for factor in (2, 4):
        other = zs[factor]
        a = reference - reference.mean(0)
        b = other - other.mean(0)
        denom = np.sqrt((a*a).sum(0) * (b*b).sum(0))
        correlation.append(np.divide((a*b).sum(0), denom, out=np.zeros(64), where=denom > 0))
    persistence = np.maximum(np.minimum(correlation[0], correlation[1]), 0)
    classes = np.unique(labels)
    means = np.stack([reference[labels == g].mean(0) for g in classes])
    variances = np.stack([reference[labels == g].var(0) for g in classes])
    rankings = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            score = (means[a] - means[b]) ** 2 / (variances[a] + variances[b] + .05)
            rankings.append(np.argsort(-score * (.05 + persistence), kind='stable'))
    keep, positions = [], [0] * len(rankings)
    while len(keep) < 32:
        for index, ranking in enumerate(rankings):
            while positions[index] < len(ranking) and int(ranking[positions[index]]) in keep:
                positions[index] += 1
            if positions[index] < len(ranking):
                keep.append(int(ranking[positions[index]]))
                positions[index] += 1
                if len(keep) == 32:
                    break
    return np.asarray(keep, dtype=np.int16)

def normalized_neighbors(grid: np.memmap, ys: np.ndarray, xs: np.ndarray,
                         mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gh, gw = grid.shape[:2]
    centers = np.asarray(grid[ys, xs], np.float32)
    if not np.isfinite(centers).all():
        raise ValueError('A selected tissue token is missing')
    z = (centers - mean) / scale
    neigh = np.empty((len(ys), 192), np.float32)
    for i, offset in enumerate((1, 2, 4)):
        group = []
        for yy, xx in ((ys - offset, xs), (ys + offset, xs),
                       (ys, xs - offset), (ys, xs + offset)):
            values = np.asarray(grid[np.clip(yy, 0, gh - 1),
                                     np.clip(xx, 0, gw - 1)], np.float32)
            bad = ~np.isfinite(values)
            if bad.any():
                values[bad] = centers[bad]
            group.append((values - mean) / scale)
        neigh[:, i * 64:(i + 1) * 64] = np.mean(group, axis=0)
    return neigh, z

def spatial_training(grid: np.memmap, eligible: np.ndarray,
                     mean: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(eligible.sum())
    n_per = len(POSITIONS) ** 2
    x = np.empty((n * n_per, 192), np.float32)
    y = np.empty((n * n_per, 64), np.float32)
    fold = np.empty(n * n_per, np.uint8)
    for i, (row, col) in enumerate(np.argwhere(eligible)):
        ys = np.repeat(row * 16 + np.asarray(POSITIONS), len(POSITIONS))
        xs = np.tile(col * 16 + np.asarray(POSITIONS), len(POSITIONS))
        part_x, part_y = normalized_neighbors(grid, ys, xs, mean, scale)
        segment = slice(i * n_per, (i + 1) * n_per)
        x[segment], y[segment] = part_x, part_y
        fold[segment] = (row >= eligible.shape[0]//2) * 2 + (col >= eligible.shape[1]//2)
    return x, y, fold

def classifier(values: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    return LogisticRegression(C=.3, class_weight='balanced', max_iter=2000,
                              random_state=430).fit(values, labels)

def align_scale(grid: np.ndarray, rows: np.ndarray, columns: np.ndarray,
                factor: int, mean: np.ndarray, scale: np.ndarray,
                keep: np.ndarray) -> np.ndarray:
    yy = (rows.astype(np.float32) + .5) / factor - .5
    xx = (columns.astype(np.float32) + .5) / factor - .5
    ymesh, xmesh = np.meshgrid(yy, xx, indexing='ij')
    aligned = np.empty((len(rows), len(columns), len(keep)), np.float32)
    for j, channel in enumerate(keep):
        values = map_coordinates(np.asarray(grid[..., channel], np.float32),
                                 [ymesh, xmesh], order=1, mode='nearest')
        values = np.where(np.isfinite(values), values, mean[channel])
        aligned[..., j] = (values - mean[channel]) / scale[channel]
    return aligned

def neighbors(grid: np.memmap, rows: np.ndarray, mean: np.ndarray,
              scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = grid.shape[:2]
    columns = np.arange(width)
    raw = np.asarray(grid[rows], np.float32)
    center = np.where(np.isfinite(raw), raw, mean)
    z = (center - mean) / scale
    result = np.empty((len(rows), width, 192), np.float32)
    for block, step in enumerate((1, 2, 4)):
        parts = []
        for yy, xx in ((np.clip(rows - step, 0, height - 1), columns),
                       (np.clip(rows + step, 0, height - 1), columns),
                       (rows, np.clip(columns - step, 0, width - 1)),
                       (rows, np.clip(columns + step, 0, width - 1))):
            values = np.asarray(grid[yy[:, None], xx[None, :]], np.float32)
            values = np.where(np.isfinite(values), values, center)
            parts.append((values - mean) / scale)
        result[..., block * 64:(block + 1) * 64] = np.mean(parts, axis=0)
    return z, result


def _inputs(manifest):
    manifest=Path(manifest).resolve();spec=json.loads(manifest.read_text())
    allowed={"format","source_frame_id","image","shape_yx","group_grid","selected_dimensions","features"}
    if spec.get("format")!="bora-wsi-response-input-v1" or set(spec)-allowed:
        raise ValueError("Use a bora-wsi-response-input-v1 manifest with predictor inputs only")
    if not isinstance(spec.get("source_frame_id"),str) or not spec["source_frame_id"].strip():
        raise ValueError("A complete source-frame identity is required")
    image,image_sha=bound_file(manifest.parent,spec["image"],"image")
    group_path,group_sha=bound_file(manifest.parent,spec["group_grid"],"group_grid")
    grid=np.load(group_path,allow_pickle=False)
    if grid.ndim!=2 or grid.dtype.kind not in "iu" or grid.min()<0 or grid.max()>255 or any(n%4 for n in grid.shape):
        raise ValueError("Use a categorical grid padded to multiples of four tiles")
    eligible=grid>0;labels=grid[eligible].astype(np.uint8);classes=np.unique(labels)
    if len(classes)<2 or not np.array_equal(classes,np.arange(1,len(classes)+1)):
        raise ValueError("At least two contiguous anonymous group IDs are required")
    dimensions=np.asarray(spec["selected_dimensions"])
    if dimensions.shape!=(64,) or dimensions.dtype.kind not in "iu" or len(np.unique(dimensions))!=64 or dimensions.min()<0 or dimensions.max()>=1280:
        raise ValueError("Declare the same 64 selected Virchow2 channel indices at every scale")
    shape=spec["shape_yx"]
    if not isinstance(shape,list) or len(shape)!=2 or any(type(n) is not int or n<1 for n in shape) or np.any(np.array(shape)>np.array(grid.shape)*224):
        raise ValueError("The tile grid must cover the complete calibrated image frame")
    arrays={};hashes={}
    if set(spec["features"])!={"1","2","4"}:raise ValueError("All three calibrated spatial scales are required")
    for factor in (1,2,4):
        path,digest=bound_file(manifest.parent,spec["features"][str(factor)],f"factor_{factor}")
        array=np.load(path,mmap_mode="r",allow_pickle=False)
        expected=tuple(np.array(grid.shape)*16//factor)+(64,)
        if array.shape!=expected or array.dtype.kind!="f":raise ValueError("Multiscale feature geometry differs")
        arrays[factor]=array;hashes[str(factor)]=digest
    info=dict(profile=PROFILE,source_frame_id=spec["source_frame_id"],source_image_sha256=image_sha,
        group_grid_sha256=group_sha,shape_yx=shape,group_count=len(classes),training_tiles=int(eligible.sum()),
        selected_dimensions=dimensions.tolist(),features_sha256=hashes,input_manifest_sha256=sha(manifest))
    return arrays,eligible,labels,info


def _coefficients(model):
    # The accepted >=3-group heads are unchanged. Binary logistic logits
    # have one sklearn row and use an explicit zero reference class.
    if len(model.classes_)==2:
        return np.vstack((np.zeros_like(model.coef_),model.coef_)).astype(np.float32),np.r_[0.,model.intercept_].astype(np.float32)
    return model.coef_.astype(np.float32),model.intercept_.astype(np.float32)


@threadpool_limits.wrap(limits=4)
def fit_slide_responses(manifest,output):
    from .full_channel_inputs import is_full_input,prepare_bora_selection
    if is_full_input(manifest):
        output=Path(output)
        if output.exists():raise FileExistsError(output)
        selected,identity=prepare_bora_selection(manifest,output.with_name(output.name+'-bora-selection'))
        result=fit_slide_responses(selected,output)
        result.update(identity,full_input_channels=1280,selection_owner='Bora',internal_selected_input_manifest=str(selected.resolve()))
        atomic(output/'GLOBAL_FIT_COMPLETE.json',result)
        return result
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    arrays,eligible,labels,info=_inputs(manifest)
    summaries={f:tile_summaries(arrays[f],f,eligible) for f in (1,2,4)}
    scalers={};standardized={}
    for factor in (1,2,4):
        mean,scale,standardized[factor]=fit_scaler(summaries[factor]);scalers[factor]=(mean,scale)
    keep=persistence_positions(standardized,labels)
    p9=classifier(np.mean(np.stack([standardized[f][:,keep] for f in (1,2,4)]),axis=0),labels)
    x,y,fold=spatial_training(arrays[1],eligible,*scalers[1]);heads=[]
    for quadrant in range(4):
        if not np.any(fold!=quadrant):raise ValueError("Neighbour reconstruction needs tissue observations in more than one spatial quadrant")
        heads.append(Ridge(alpha=100).fit(x[fold!=quadrant],y[fold!=quadrant]))
    completed=np.empty_like(y)
    for quadrant in range(4):
        indices=np.flatnonzero(fold==quadrant)
        if len(indices):completed[indices]=heads[quadrant].predict(x[indices])
    predicted=completed.reshape(len(labels),len(POSITIONS)**2,64).mean(1)
    p19=classifier(predicted,labels);c9,i9=_coefficients(p9);c19,i19=_coefficients(p19)
    state=dict(selected_dimensions=np.asarray(info["selected_dimensions"],np.int16),persistent_positions=keep,
        classes=p9.classes_.astype(np.uint8),p9_coef=c9,p9_intercept=i9,p19_coef=c19,p19_intercept=i19,
        ridge_coef=np.stack([h.coef_ for h in heads]).astype(np.float32),
        ridge_intercept=np.stack([h.intercept_ for h in heads]).astype(np.float32),
        ridge_global_coef=np.mean(np.stack([h.coef_ for h in heads]),axis=0).astype(np.float32),
        ridge_global_intercept=np.mean(np.stack([h.intercept_ for h in heads]),axis=0).astype(np.float32))
    for factor in (1,2,4):state[f"mean_{factor}"],state[f"scale_{factor}"]=scalers[factor]
    output.mkdir(parents=True);path=output/"slide_global_rgb9plus19_state.npz"
    np.savez_compressed(path,**state)
    result=dict(info,status="COMPLETE",format="bora-slide-global-fit-v1",model_sha256=sha(path),
        class_ids=p9.classes_.tolist(),spatial_training_samples=len(y),
        one_fit_used_across_slide=True,independently_fitted_windows=False,
        group_response_heads="balanced logistic C=0.3; multiscale persistence and four-quadrant ridge alpha=100",
        tissue_identity_classifier="separate Neighbor context fastPLS model")
    atomic(output/"GLOBAL_FIT_COMPLETE.json",result);return result


@threadpool_limits.wrap(limits=4)
def project_slide_responses(manifest,fit_folder,output):
    from .full_channel_inputs import is_full_input,read_full_input
    if is_full_input(manifest):
        _,_,_,_,identity=read_full_input(manifest)
        receipt=json.loads((Path(fit_folder)/'GLOBAL_FIT_COMPLETE.json').read_text())
        for key,value in identity.items():
            if receipt.get(key)!=value:raise ValueError('Full-channel input differs from the fitted Bora input')
        selected=Path(receipt['internal_selected_input_manifest'])
        audit=json.loads((selected.parent/'BORA_SELECTION_COMPLETE.json').read_text())
        if audit['selected_input_manifest_sha256']!=sha(selected):raise ValueError('Internal Bora selection changed')
        result=project_slide_responses(selected,fit_folder,output)
        result.update(identity,full_input_channels=1280,selection_owner='Bora')
        atomic(Path(output)/'GLOBAL_PROJECTION_COMPLETE.json',result)
        return result
    output=Path(output);fit_folder=Path(fit_folder)
    if output.exists():raise FileExistsError(output)
    arrays,eligible,labels,info=_inputs(manifest)
    receipt=json.loads((fit_folder/"GLOBAL_FIT_COMPLETE.json").read_text());model=fit_folder/"slide_global_rgb9plus19_state.npz"
    if receipt.get("status")!="COMPLETE" or sha(model)!=receipt["model_sha256"]:
        raise ValueError("The slide-wide fitted model is not frozen")
    for key in ("source_frame_id","source_image_sha256","group_grid_sha256","features_sha256","selected_dimensions","shape_yx"):
        if receipt.get(key)!=info[key]:raise ValueError("Fitted responses belong to a different image, grouping or feature selection")
    with np.load(model,allow_pickle=False) as source:state={k:source[k] for k in source.files}
    k=info["group_count"];height,width=map(int,np.array(eligible.shape)*16)
    if (state["classes"].tolist()!=list(range(1,k+1)) or state["p9_coef"].shape!=(k,32)
            or state["p19_coef"].shape!=(k,64) or state["ridge_global_coef"].shape!=(64,192)):
        raise ValueError("Frozen response model dimensions differ")
    output.mkdir(parents=True)
    probability=np.lib.format.open_memmap(output/"group-probabilities-token-grid.f16.npy",mode="w+",dtype=np.float16,shape=(height,width,k))
    color=np.lib.format.open_memmap(output/"rgb9plus19-token-grid.u8.npy",mode="w+",dtype=np.uint8,shape=(height,width,3))
    palette=np.array([tuple(round(255*c) for c in colorsys.hsv_to_rgb((i*.61803398875)%1,.68,.9)) for i in range(k)],np.float32)
    # The two smaller lattices are materialized exactly as in the accepted run.
    arrays[2]=np.asarray(arrays[2],np.float32);arrays[4]=np.asarray(arrays[4],np.float32)
    columns=np.arange(width);keep=state["persistent_positions"].astype(int)
    projected_tokens=0;unlabelled_projected_tokens=0
    for start in range(0,height,16):
        rows=np.arange(start,min(start+16,height))
        z,neighbour_x=neighbors(arrays[1],rows,state["mean_1"],state["scale_1"])
        aligned=[align_scale(arrays[f],rows,columns,f,state[f"mean_{f}"],state[f"scale_{f}"],keep) for f in (2,4)]
        consensus=(z[...,keep]+aligned[0]+aligned[1])/3
        p9=softmax(consensus.reshape(-1,32)@state["p9_coef"].T+state["p9_intercept"],axis=1).reshape(len(rows),width,k)
        reconstructed=(neighbour_x.reshape(-1,192)@state["ridge_global_coef"].T+state["ridge_global_intercept"]).reshape(len(rows),width,64)
        p19=softmax(reconstructed.reshape(-1,64)@state["p19_coef"].T+state["p19_intercept"],axis=1).reshape(len(rows),width,k)
        # Labels identify fitting observations. They do not delimit where a
        # fitted head can be applied: Visium has tissue outside measured spots.
        # Unencoded glass is represented by nonfinite base-scale features.
        p=(p9+p19)/2
        support=np.isfinite(np.asarray(arrays[1][rows],np.float32)).all(axis=2)
        labelled=eligible[rows[:,None]//16,columns[None,:]//16]
        projected_tokens+=int(support.sum())
        unlabelled_projected_tokens+=int(np.count_nonzero(support&~labelled))
        p[~support]=0
        probability[rows]=p.astype(np.float16)
        rgb=np.rint(np.clip(p@palette,0,255)).astype(np.uint8);rgb[~support]=255;color[rows]=rgb
        probability.flush();color.flush()
    import pyvips
    raw=np.asarray(color).tobytes();image=pyvips.Image.new_from_memory(raw,width,height,3,"uchar")
    h,w=info["shape_yx"];image.resize(14,kernel="linear").crop(0,0,w,h).tiffsave(
        str(output/"rgb9plus19-fullslide-0p5um.tif"),tile=True,pyramid=True,bigtiff=True,compression="jpeg",Q=88,xres=2000,yres=2000)
    result=dict(info,status="COMPLETE",format=RESPONSE_FORMAT,model_sha256=receipt["model_sha256"],
        probability_sha256=sha(output/"group-probabilities-token-grid.f16.npy"),
        shape_yxc=[int(height),int(width),k],class_ids=list(range(1,k+1)),
        projection_support="finite base-scale Virchow2 tokens, independent of fitting labels",
        projected_tokens=projected_tokens,unlabelled_projected_tokens=unlabelled_projected_tokens,
        one_fit_used_across_slide=True,independently_fitted_windows=False,
        fit_receipt_sha256=sha(fit_folder/"GLOBAL_FIT_COMPLETE.json"))
    atomic(output/"GLOBAL_PROJECTION_COMPLETE.json",result);return result
