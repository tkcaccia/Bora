"""Descriptive tile-level contrasts for every class, without pixel p-values."""
import csv
import numpy as np
from sklearn.metrics import roc_auc_score
from .common import save_json

def write_feature_report(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scores=np.zeros((5,len(ctx.selected)))
    for group in range(1,6):
        for channel in range(len(ctx.selected)):
            auc=[]
            for opponent in range(1,6):
                if opponent==group:continue
                use=(ctx.labels==group)|(ctx.labels==opponent)
                value=roc_auc_score(ctx.labels[use]==group,ctx.tile_z[use,channel]);auc.append(max(value,1-value))
            scores[group-1,channel]=min(auc)
    selected=[np.argsort(-row,kind="stable")[:3] for row in scores]
    union=sorted(set(int(i) for row in selected for i in row))
    data={"definition":"minimum direction-free class-versus-each-other-class AUC, tile observations only", "scope":"descriptive within-ROI; selected pool of 64 dimensions; no significance/generalization claim",
        "by_class":{str(g):{"name":ctx.info["compact_class_names"][str(g)],"dimensions":ctx.selected[selected[g-1]].tolist(),"weakest_pair_auc":scores[g-1,selected[g-1]].tolist()} for g in range(1,6)}}
    save_json(ctx.root/"best_features_by_class.json",data)
    with (ctx.root/"feature_contrasts.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["class","name","dimension","minimum_pairwise_direction_free_auc","selected_top3_for_class"])
        for g in range(1,6):
            for c,dimension in enumerate(ctx.selected):w.writerow([g,ctx.info["compact_class_names"][str(g)],int(dimension),scores[g-1,c],c in selected[g-1]])
    fig,ax=plt.subplots(figsize=(max(9,len(union)*.65),4.5),constrained_layout=True)
    plot=ax.imshow(scores[:,union],vmin=.5,vmax=1,cmap="viridis",aspect="auto")
    ax.set_yticks(range(5),[ctx.info["compact_class_names"][str(g)] for g in range(1,6)])
    ax.set_xticks(range(len(union)),[str(ctx.selected[c]) for c in union],rotation=45)
    for g in range(5):
        for i,c in enumerate(union):ax.text(i,g,f"{scores[g,c]:.2f}"+("*" if c in selected[g] else ""),ha="center",va="center",fontsize=8,color="white" if scores[g,c]<.8 else "black")
    ax.set_xlabel("Virchow2 dimension ID (not a biological marker)")
    ax.set_title("Evidence for EACH tissue class: weakest pairwise discrimination\n* best three within the selected pool; 169 coarse tile observations, descriptive only")
    fig.colorbar(plot,ax=ax,label="Minimum direction-free AUC");fig.savefig(ctx.root/"features_by_class.png",dpi=180);plt.close(fig)
