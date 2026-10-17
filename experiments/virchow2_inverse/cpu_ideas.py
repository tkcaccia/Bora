"""Run the seventeen non-inversion hypotheses without opening fine annotations."""
from __future__ import annotations
import argparse,gc,json,time,traceback
from dataclasses import asdict
from pathlib import Path
from itertools import combinations
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from sklearn.decomposition import NMF
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .common import coarse_tile_observations,save_json,digest,pad_image
from .selection import round_robin
from .protocol import Context,expand,rgb_scale,probability_rgb,train_on_map,coarse_summaries,bora_rgb,fit_classifier,TITLES
from .algorithms import graph_transport,pair_experts,feature_gradient,barrier_refine,superpixel_probabilities,object_maps,soft_transport,topology_project,quilt

def grid_prob(z,ctx):
    summaries,y,_,_=coarse_tile_observations(z,ctx.coarse)
    clf=fit_classifier(summaries,y)
    return clf.predict_proba(z.reshape(-1,z.shape[-1])).reshape(*z.shape[:2],5).astype(np.float32)

def rgb_prob_grid(p,ctx):return probability_rgb(expand(p,ctx.shape))

def balanced_features(values,labels,penalty=None,k=32):
    means=[values[labels==g].mean(0) for g in range(1,6)];variances=[values[labels==g].var(0) for g in range(1,6)]
    lists=[]
    for a,b in combinations(range(5),2):
        score=(means[a]-means[b])**2/(variances[a]+variances[b]+.05)
        if penalty is not None:score=score*penalty
        lists.append(np.argsort(-score))
    return round_robin(lists,min(k,values.shape[1]))


def additional_grid(ctx,name):
    meta=next(r for r in json.loads((ctx.root/"additional_inference/manifest.json").read_text()) if r["name"]==name)
    values=np.load(ctx.root/f"additional_inference/{name}.npy").astype(np.float32)
    if name.startswith("stain"):return values,meta
    sh,sw=meta["shape_yx"];h,w=ctx.shape
    y=(np.arange(ctx.grid[0])+.5)*14*sh/h/14-.5
    x=(np.arange(ctx.grid[1])+.5)*14*sw/w/14-.5
    yy,xx=np.meshgrid(y,x,indexing="ij")
    aligned=np.stack([ndi.map_coordinates(values[...,i],[yy,xx],order=1,mode="nearest") for i in range(values.shape[-1])],-1)
    return aligned,meta

def scale_z(ctx,name):
    values,meta=additional_grid(ctx,name)
    summaries,y,_,_=coarse_tile_observations(values,ctx.coarse)
    scaler=StandardScaler().fit(summaries);scaler.scale_=np.maximum(scaler.scale_,.05)
    return scaler.transform(values.reshape(-1,values.shape[-1])).reshape(values.shape).astype(np.float32),scaler.transform(summaries),meta

def run_idea(n,c):
    z=c.z;shape=c.shape;root=c.root;extras={};prediction=None
    details={"preset":"one fixed exploratory preset, no fine-mask selection","output_type":"feature RGB","original_rgb_supplies_detail":False}
    if n==6:
        rgb,prediction,extras,selected=pair_experts(c)
        details.update(pair_dimensions=selected,wrapper="pair-restricted Bora wand, fixed nearest-interface ownership; tied junctions protected")
    elif n==7:
        padded=pad_image(c.image).astype(float)/255;gh,gw=c.grid
        colour=padded.reshape(gh,14,gw,14,3).mean((1,3));od=-np.log(np.maximum(colour,.005))
        nuisance=np.concatenate([colour,od,od**2],-1).astype(np.float32)
        reg=Ridge(alpha=10).fit(nuisance.reshape(-1,9),z.reshape(-1,z.shape[2]))
        residual=z-reg.predict(nuisance.reshape(-1,9)).reshape(z.shape)
        summaries,y,_,_=coarse_tile_observations(residual,c.coarse);scaler=StandardScaler().fit(summaries)
        scaler.scale_=np.maximum(scaler.scale_,.05);residual=scaler.transform(residual.reshape(-1,z.shape[2])).reshape(z.shape)
        rgb=rgb_prob_grid(grid_prob(residual,c),c);extras={"residual_features":residual.astype(np.float32),"nuisance_coefficients":reg.coef_.astype(np.float32)}
        details.update(nuisance="patch mean RGB, optical density and squared optical density",original_rgb_used="estimate the colour-explained feature subspace",residual_variance_fraction=float(np.mean(residual.var((0,1)))))
    elif n==9:
        middle,mt,_=scale_z(c,"scale_1");wide,wt,_=scale_z(c,"scale_2")
        correlations=np.stack([np.array([np.corrcoef(c.tile_z[:,i],other[:,i])[0,1] for i in range(z.shape[2])]) for other in [mt,wt]])
        persistence=np.maximum(np.nan_to_num(correlations.min(0)),0)
        keep=balanced_features(c.tile_z,c.labels,.05+persistence,32)
        consensus=(z[...,keep]+middle[...,keep]+wide[...,keep])/3
        rgb=rgb_prob_grid(grid_prob(consensus,c),c)
        details.update(selected_dimensions=c.selected[keep].tolist(),cross_scale_correlation=correlations.tolist(),effective_sampling_um=[.5,1,2])
        extras={"scale_consensus":consensus.astype(np.float32)}
    elif n==10:
        middle,_,_=scale_z(c,"scale_1");wide,_,_=scale_z(c,"scale_2")
        p1,p2=grid_prob(middle,c),grid_prob(wide,c);context=np.sqrt(p1*p2)
        top=np.argsort(context,axis=2)[...,-2:];gate=np.full_like(context,.15)
        np.put_along_axis(gate,top,1,axis=2)
        p=c.p*np.sqrt(context+1e-5)*gate;p/=p.sum(2,keepdims=True)
        rgb=rgb_prob_grid(p,c);extras={"context_probabilities":context,"local_probabilities":c.p}
        details.update(context_gate="two strongest wider-context classes have full weight; other classes 0.15",context_um_per_window=[224,448],fine_window_um=112)
    elif n==11:
        var=np.load(root/"features/overlap_view_variance.npy",mmap_mode="r")[...,c.selected].astype(np.float32)
        variation=np.mean(var/(c.scaler.scale_**2),axis=2)
        counts=np.zeros(c.grid,int)
        for y,x in c.inference["window_coordinates_yx"]:counts[y//14:y//14+16,x//14:x//14+16]+=1
        multi=counts>=2;threshold=float(np.median(variation[multi]));trust=(variation<=threshold)&multi
        native=expand(trust.astype(np.float32),shape)>=.5
        feature_prediction=bora_rgb(c.feature_rgb,c.coarse)
        prediction=np.where(native,feature_prediction,c.baseline).astype(np.uint8)
        rgb=np.where(native[...,None],c.feature_rgb,c.image)
        extras={"context_variation":expand(variation,shape),"trust_mask":(native*255).astype(np.uint8),"context_count":counts}
        details.update(output_type="hybrid decision and display",original_rgb_supplies_detail=True,variation_threshold=threshold,trusted_fraction=float(native.mean()),calibrated_uncertainty=False)
    elif n==12:
        p=graph_transport(z,c.p,c.token_labels,c.token_seeds)
        rgb=rgb_prob_grid(p,c);extras={"transported_probabilities":p}
        details.update(graph="four spatial neighbours; exponential selected-feature edge weights; coarse interior seeds; harmonic transport")
    elif n==13:
        energy=expand(feature_gradient(z),shape);rgb=rgb_scale(energy)
        prediction=barrier_refine(c,energy)
        gray=cv2.cvtColor(c.image,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
        smooth=ndi.gaussian_filter(gray,2);gy,gx=np.gradient(smooth)
        he_energy=np.sqrt(gx**2+gy**2);he_energy/=max(float(np.quantile(he_energy,.95)),1e-6)
        energy_scale=max(float(np.quantile(energy,.95)),1e-6)
        # Match scale to the H&E barrier control before watershed compactness acts.
        prediction=barrier_refine(c,energy/energy_scale)
        control=barrier_refine(c,he_energy)
        import tifffile
        tifffile.imwrite(root/"control_he_barrier.tif",control,compression="deflate")
        extras={"semantic_barrier":energy.astype(np.float32)}
        details.update(output_type="boundary barrier",wrapper="seeded watershed inside the same Bora editable band; coarse interiors fixed",matched_control="control_he_barrier.tif")
    elif n==14:
        p,segments=superpixel_probabilities(c.image,c.native_p)
        rgb=probability_rgb(p);extras={"superpixels":segments.astype(np.int32)}
        details.update(original_rgb_supplies_detail=True,original_rgb_used="SLIC boundaries and graph colour similarity",superpixel_count=int(segments.max()+1))
    elif n==15:
        objects,nuclei,lumens,threshold=object_maps(c.image,c.coarse)
        a=np.concatenate([c.native_p,objects],-1)
        scaler=StandardScaler().fit(coarse_summaries(a,c.records));scaler.scale_=np.maximum(scaler.scale_,1e-4)
        a=scaler.transform(a.reshape(-1,a.shape[2])).reshape(a.shape).astype(np.float32)
        p,clf=train_on_map(a,c);rgb=probability_rgb(p)
        extras={"candidate_nuclei":(nuclei*255).astype(np.uint8),"candidate_lumens":(lumens*255).astype(np.uint8),"object_relations":objects}
        details.update(original_rgb_supplies_detail=True,candidate_object_annotations=False,hematoxylin_threshold=threshold,
                       variables=["five class evidences","nuclear density 6um","nuclear density 20um","lumen neighbourhood 12um","distance to lumen","nuclear orientation coherence"],coefficients=clf.coef_.tolist())
    elif n==16:
        p,record=soft_transport(z,c.tile_z,c.labels)
        rgb=rgb_prob_grid(p,c);extras={"tissue_mixtures":p}
        sensitivity=[]
        for strength in (0,.5):
            alternative,info=soft_transport(z,c.tile_z,c.labels,strength)
            sensitivity.append({**info,"mean_absolute_probability_change":float(np.mean(abs(alternative-p)))})
        details.update(**record,prior_sensitivity=sensitivity,mixtures_are_estimated_not_measured=True)
    elif n==17:
        preliminary=bora_rgb(c.feature_rgb,c.coarse)
        prediction,record=topology_project(c.coarse,preliminary,c.native_p)
        rgb=c.feature_rgb;extras={"topology_change":((prediction!=preliminary)*255).astype(np.uint8)}
        details.update(**record,wrapper="feature Bora proposal, local Euler-change penalty, strong evidence may override topology; topology is not fixed to coarse")
    elif n==18:
        rgb,donors,source_yx,records=quilt(c);extras={"donor_map":donors,"donor_pixel_yx":source_yx}
        if not np.array_equal(rgb,c.image[source_yx[...,0],source_yx[...,1]]):raise AssertionError("Quilt pixels differ from recorded real donors")
        save_json(root/"18/donors.json",records)
        details.update(output_type="retrieved H&E mosaic",original_rgb_supplies_detail=True,original_rgb_used="donor patches only",patch_px=42,nominal_stride_px=28,
                       matching="eight nearest selected-feature donors, feature cost plus overlap cost, minimum-error seams",all_encoder_context_boxes_disjoint=True,donors=len(records))
    elif n==19:
        neighbours=np.concatenate([(np.roll(z,k,axis=0)+np.roll(z,-k,axis=0)+np.roll(z,k,axis=1)+np.roll(z,-k,axis=1))/4 for k in (1,2,4)],-1)
        # Reflection replaces rolled borders, preventing distant edge neighbours.
        for k in (1,2,4):
            part=neighbours[...,((1,2,4).index(k))*z.shape[2]:((1,2,4).index(k)+1)*z.shape[2]]
            padded=np.pad(z,((k,k),(k,k),(0,0)),mode="reflect")
            part[:]=(padded[:-2*k,k:-k]+padded[2*k:,k:-k]+padded[k:-k,:-2*k]+padded[k:-k,2*k:])/4
        yy,xx=np.indices(c.grid);folds=(yy>=c.grid[0]//2)*2+(xx>=c.grid[1]//2)
        completed=np.empty_like(z);records=[]
        for fold in range(4):
            test=folds==fold;train=~ndi.binary_dilation(test,iterations=4)
            model=Ridge(alpha=100).fit(neighbours[train],z[train]);completed[test]=model.predict(neighbours[test])
            records.append({"fold":fold,"train":int(train.sum()),"test":int(test.sum()),"training_buffer_tokens":4})
        rgb=rgb_prob_grid(grid_prob(completed,c),c);extras={"completed_features":completed,"completion_residual":np.sqrt(np.mean((completed-z)**2,axis=2))}
        details.update(protocol="single token masked; neighbours at 1,2,4 token offsets; four spatial quadrant fits, 4-token training exclusion buffer",folds=records,
                       own_token_in_predictor=False,encoder_context_shared_between_tokens=True)
    elif n==20:
        import pywt
        barrier=expand(feature_gradient(z),shape);gate=np.clip(barrier/max(float(np.quantile(barrier,.9)),1e-6),0,1)
        reconstructed=[]
        for ch in range(3):
            coeff=pywt.wavedec2(c.image[...,ch].astype(float),"haar",level=3)
            changed=[coeff[0]]
            for detail in coeff[1:]:
                g=cv2.resize(gate,(detail[0].shape[1],detail[0].shape[0]),interpolation=cv2.INTER_AREA)
                changed.append(tuple(a*(.25+.75*g) for a in detail))
            reconstructed.append(pywt.waverec2(changed,"haar")[:shape[0],:shape[1]])
        rgb=np.rint(np.stack(reconstructed,-1)).clip(0,255).astype(np.uint8);extras={"detail_retention":(.25+.75*gate).astype(np.float32)}
        details.update(output_type="hybrid H&E reconstruction",original_rgb_supplies_detail=True,wavelet="Haar, 3 levels",detail_weight_range=[.25,1.])
    else:raise ValueError(n)
    return c.output(n,rgb,details,prediction,extras)

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--ids",type=int,nargs="+",default=list([6,7,9,10,11,12,13,14,15,16,17,18,19,20]));a=p.parse_args()
    c=Context(a.source,a.root)
    Image.fromarray(c.image).save(a.root/"original.png")
    import tifffile
    tifffile.imwrite(a.root/"coarse.tif",c.coarse,compression="deflate")
    tifffile.imwrite(a.root/"control_he_bora.tif",c.baseline,compression="deflate")
    tifffile.imwrite(a.root/"control_feature_bora.tif",bora_rgb(c.feature_rgb,c.coarse),compression="deflate")
    Image.fromarray(c.feature_rgb).save(a.root/"control_feature_rgb.png")
    from bora.boundary import BoundaryConfig
    save_json(a.root/"protocol.json",{"sample":c.info["sample"],"class_names":c.info["compact_class_names"],"native_mpp_xy":[.5,.5],
        "optical_resolution_um":None,"objective_magnification":None,"native_type":"ROI, not WSI","token_fov_um_xy":[7,7],
        "boundary_config":asdict(BoundaryConfig()),"fine_reference_read_during_prediction":False,"preset_frozen_before_scoring":True,
        "coarse_labels_are_reference_derived":True,"feature_selection":c.selection,"selected_dimensions":c.selected.tolist(),
        "scope":"single annotated ROI; exploratory controlled boundary reconstruction, not external patient validation"})
    source_hashes={name:digest(Path(__file__).with_name(name)) for name in ["cpu_ideas.py","algorithms.py","protocol.py","gpu_ideas.py"]}
    with threadpool_limits(limits=2):
        for n in a.ids:
            start=time.perf_counter();folder=a.root/f"{n:02d}";folder.mkdir(exist_ok=True)
            try:
                print(json.dumps({"idea":n,"title":TITLES[n-1],"starting":True}),flush=True)
                result=run_idea(n,c);result.update(seconds=time.perf_counter()-start,source_sha256=source_hashes)
                save_json(folder/"result.json",result);print(json.dumps({"idea":n,"completed":True,"seconds":result["seconds"]}),flush=True)
            except Exception as e:
                save_json(folder/"result.json",{"id":n,"title":TITLES[n-1],"status":"failed","error":str(e),"traceback":traceback.format_exc(),"source_sha256":source_hashes})
                print(json.dumps({"idea":n,"failed":True,"error":str(e)}),flush=True)
            gc.collect()
    print(json.dumps({"cpu_run_finished":True}),flush=True)

if __name__=="__main__":main()
