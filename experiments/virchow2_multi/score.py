"""Score frozen native masks, then rank paired cases with equal dataset weight."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from experiments.virchow2_inverse.common import digest,save_json
from .methods import METHODS


def class_boundary(labels,group,valid):
    foreground=labels==group;edge=np.zeros(labels.shape,bool)
    d=(foreground[1:]!=foreground[:-1])&valid[1:]&valid[:-1];edge[1:]|=d;edge[:-1]|=d
    d=(foreground[:,1:]!=foreground[:,:-1])&valid[:,1:]&valid[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
    return edge


class Evaluator:
    def __init__(self,truth,mpp_xy):
        self.truth=truth;self.valid=truth>0;self.classes=np.unique(truth[self.valid]);self.spacing=np.array(mpp_xy[::-1])
        self.points={int(g):np.argwhere(class_boundary(truth,g,self.valid))*self.spacing for g in self.classes}
        self.trees={g:cKDTree(p) if len(p) else None for g,p in self.points.items()}

    def score(self,prediction):
        if prediction.shape!=self.truth.shape:raise ValueError("Native geometry mismatch")
        result={};distances=[];bf={str(t):[] for t in (4,8,16)}
        for g in self.classes:
            a=(prediction==g)&self.valid;b=self.truth==g
            intersection=int((a&b).sum());union=int((a|b).sum());total=int(a.sum()+b.sum())
            row=dict(iou=intersection/union,dice=2*intersection/total,truth_pixels=int(b.sum()),prediction_pixels=int(a.sum()))
            points=np.argwhere(class_boundary(prediction,g,self.valid))*self.spacing;target=self.points[int(g)]
            if len(target):
                if len(points):
                    dp=self.trees[int(g)].query(points,workers=2)[0];dt=cKDTree(points).query(target,workers=2)[0]
                    distances.extend([dp,dt])
                    for tolerance in (4,8,16):
                        precision=float(np.mean(dp<=tolerance));recall=float(np.mean(dt<=tolerance))
                        bf[str(tolerance)].append(2*precision*recall/(precision+recall) if precision+recall else 0.)
                else:
                    for tolerance in (4,8,16):bf[str(tolerance)].append(0.)
            result[str(int(g))]=row
        return dict(macro_iou=float(np.mean([r["iou"] for r in result.values()])),macro_dice=float(np.mean([r["dice"] for r in result.values()])),
            worst_class_iou=min(r["iou"] for r in result.values()),pixel_accuracy=float(np.mean(prediction[self.valid]==self.truth[self.valid])),
            boundary_f1_at_um={k:float(np.mean(v)) if v else None for k,v in bf.items()},
            matched_boundary_hd95_um=float(np.percentile(np.concatenate(distances),95)) if distances else None,
            boundary_caveat="Class-specific boundary F1; ignore interfaces excluded. HD95 omits missing predicted contours, so do not use HD95 alone.",
            valid_pixels=int(self.valid.sum()),per_class=result,ignore_code=0)


def ranking(cases,bootstrap=3000):
    datasets=sorted(set(c["dataset"] for c in cases));rows=[];rng=np.random.default_rng(7730)
    # Identical case resamples for all methods preserve pairing to H&E.
    indices={d:rng.integers(0,len([c for c in cases if c["dataset"]==d]),size=(bootstrap,len([c for c in cases if c["dataset"]==d]))) for d in datasets}
    for method,title in METHODS.items():
        key=f"{method:02d}";per_dataset={};draws=[]
        all_after=[];all_before=[];all_delta=[]
        for dataset in datasets:
            subset=[c for c in cases if c["dataset"]==dataset]
            after=np.array([c["methods"][key]["after"]["macro_iou"] for c in subset])
            before=np.array([c["methods"][key]["before"]["macro_iou"] for c in subset])
            control=np.array([c["controls"]["he"]["macro_iou"] for c in subset]);delta=after-control
            per_dataset[dataset]=dict(cases=len(subset),macro_iou=float(after.mean()),before_macro_iou=float(before.mean()),
                delta_vs_he=float(delta.mean()),wins_vs_he=int(np.sum(delta>1e-8)),ties_vs_he=int(np.sum(abs(delta)<=1e-8)),
                median_iou=float(np.median(after)),mean_rank=float(np.mean([c["methods"][key]["rank"] for c in subset])))
            draws.append(delta[indices[dataset]].mean(1));all_after.extend(after);all_before.extend(before);all_delta.extend(delta)
        interval=np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist()
        expert_only=[v for d,v in per_dataset.items() if d!="camelyon16-subset"]
        rows.append(dict(id=method,title=title,cases=len(cases),dataset_balanced_macro_iou=float(np.mean([v["macro_iou"] for v in per_dataset.values()])),
            dataset_balanced_before_iou=float(np.mean([v["before_macro_iou"] for v in per_dataset.values()])),
            dataset_balanced_delta_vs_he=float(np.mean([v["delta_vs_he"] for v in per_dataset.values()])),
            paired_delta_bootstrap95=interval,expert_tissue_dataset_balanced_iou=float(np.mean([v["macro_iou"] for v in expert_only])),
            image_mean_iou=float(np.mean(all_after)),image_median_iou=float(np.median(all_after)),
            wins_vs_he=int(np.sum(np.array(all_delta)>1e-8)),losses_vs_he=int(np.sum(np.array(all_delta)<-1e-8)),
            ties_vs_he=int(np.sum(abs(np.array(all_delta))<=1e-8)),mean_case_rank=float(np.mean([c["methods"][key]["rank"] for c in cases])),
            per_dataset=per_dataset))
    rows.sort(key=lambda r:(-r["dataset_balanced_macro_iou"],r["id"]))
    for index,row in enumerate(rows):row["rank"]=index+1
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args()
    frozen=json.loads((a.root/"ALL_PREDICTIONS_FROZEN.json").read_text());plan=json.loads((a.root/"PLAN.json").read_text())
    # Verify EVERY prediction before opening ANY reference.
    for sample in plan["samples"]:
        folder=a.root/"cases"/sample["case"];path=folder/"FROZEN.json"
        if digest(path)!=frozen["case_manifest_sha256"][sample["case"]]:raise ValueError("Case manifest changed")
        for name,sha in json.loads(path.read_text())["files_sha256"].items():
            if digest(folder/name)!=sha:raise ValueError("Frozen prediction changed")
    cases=[];csv_rows=[]
    for sample in plan["samples"]:
        folder=a.root/"cases"/sample["case"]
        if digest(folder/"reference.native.tif")!=sample["reference_native_sha256"]:raise ValueError("Reference changed")
        truth=tifffile.imread(folder/"reference.native.tif");evaluator=Evaluator(truth,sample["native_mpp_xy"])
        coarse=tifffile.imread(folder/"coarse.tif");coarse_native=cv2.resize(coarse,(truth.shape[1],truth.shape[0]),interpolation=cv2.INTER_NEAREST)
        controls={"coarse":evaluator.score(coarse_native)}
        for name in ("he","features"):controls[name]=evaluator.score(tifffile.imread(folder/name/"bora.native.tif"))
        methods={}
        for number,title in METHODS.items():
            key=f"{number:02d}";before=evaluator.score(tifffile.imread(folder/key/"before.native.tif"));after=evaluator.score(tifffile.imread(folder/key/"bora.native.tif"))
            methods[key]=dict(id=number,title=title,before=before,after=after,delta_vs_he=after["macro_iou"]-controls["he"]["macro_iou"],delta_vs_before=after["macro_iou"]-before["macro_iou"])
        values=sorted(methods,key=lambda k:(-methods[k]["after"]["macro_iou"],int(k)))
        for key in values:
            methods[key]["rank"]=1+sum(methods[other]["after"]["macro_iou"]>methods[key]["after"]["macro_iou"]+1e-12 for other in values)
            row=methods[key];csv_rows.append(dict(case=sample["case"],dataset=sample["dataset"],group=sample["group"],method=row["id"],rank=row["rank"],
                macro_iou=row["after"]["macro_iou"],macro_dice=row["after"]["macro_dice"],before_iou=row["before"]["macro_iou"],
                delta_vs_before=row["delta_vs_before"],he_iou=controls["he"]["macro_iou"],delta_vs_he=row["delta_vs_he"],
                worst_class_iou=row["after"]["worst_class_iou"],boundary_f1_8um=row["after"]["boundary_f1_at_um"]["8"]))
        represented=set(np.unique(coarse))-{0};missing=sorted(set(np.unique(truth))-{0}-represented)
        row=dict(case=sample["case"],dataset=sample["dataset"],group=sample["group"],sample=sample,controls=controls,methods=methods,
            missing_coarse_classes=list(map(int,missing)),coarse_class_coverage=float(np.mean(np.isin(truth[truth>0],list(represented)))))
        cases.append(row);save_json(folder/"scores.json",row)
        print(json.dumps(dict(scored=sample["case"],best=values[0],best_iou=methods[values[0]]["after"]["macro_iou"],he_iou=controls["he"]["macro_iou"])),flush=True)
    ranks=ranking(cases)
    save_json(a.root/"results.json",dict(cases=cases,ranking=ranks,metric="native-reference per-image macro IoU; equal-weight dataset means for primary rank",
        uncertainty="3000 paired bootstrap resamples within dataset; descriptive, not a prospective clinical efficacy interval",
        scoring_source_sha256=digest(__file__)))
    with (a.root/"scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    with (a.root/"ranking.csv").open("w",newline="") as f:
        fields=["rank","id","title","cases","dataset_balanced_macro_iou","dataset_balanced_before_iou","dataset_balanced_delta_vs_he","image_mean_iou","wins_vs_he","losses_vs_he","ties_vs_he","mean_case_rank"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(ranks)

if __name__=="__main__":main()
