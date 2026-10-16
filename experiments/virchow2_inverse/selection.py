"""Class-aware selection of frozen embedding dimensions, never fine masks."""
from __future__ import annotations

from itertools import combinations
import warnings

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def round_robin(ranked_lists, n_features):
    selected=[];positions=[0]*len(ranked_lists)
    while len(selected)<n_features:
        changed=False
        for i,ranking in enumerate(ranked_lists):
            while positions[i]<len(ranking) and int(ranking[positions[i]]) in selected:positions[i]+=1
            if positions[i]<len(ranking):
                selected.append(int(ranking[positions[i]]));positions[i]+=1;changed=True
                if len(selected)==n_features:break
        if not changed:break
    return selected


def select_features(values,labels,method,top_k=64,seed=430):
    values=np.asarray(values,dtype=np.float64);labels=np.asarray(labels)
    if values.ndim!=2 or not np.isfinite(values).all() or len(values)!=len(labels):raise ValueError("Finite aligned tile summaries required")
    classes=np.unique(labels)
    if len(classes)<2:raise ValueError("At least two coarse classes required")
    scaler=StandardScaler().fit(values);z=scaler.transform(values)
    means=np.stack([z[labels==g].mean(0) for g in classes])
    variances=np.stack([z[labels==g].var(0) for g in classes])
    pairs=list(combinations(range(len(classes)),2))
    fisher=np.array([(means[a]-means[b])**2/(variances[a]+variances[b]+.01) for a,b in pairs])
    ranks=[np.argsort(-row,kind="stable") for row in fisher]
    top_k=min(top_k,values.shape[1]);weights=fisher.max(0)
    if method=="pairwise_fisher":
        chosen=round_robin(ranks,top_k)
        rule="Round-robin strongest Fisher contrasts for every coarse class pair; denominator var_a+var_b+0.01 after unit scaling"
    elif method=="class_l1":
        estimator=OneVsRestClassifier(LogisticRegression(penalty="l1",solver="liblinear",C=1.,class_weight="balanced",max_iter=2500,random_state=seed),n_jobs=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",FutureWarning);estimator.fit(z,labels)
        coefficients=np.stack([np.abs(e.coef_[0]) for e in estimator.estimators_])
        weights=coefficients.max(0)
        nonzero=np.flatnonzero(weights>1e-8)
        if not len(nonzero):raise ValueError("L1 selector fitted no active features")
        class_ranks=[np.array([i for i in np.argsort(-row,kind="stable") if weights[i]>1e-8]) for row in coefficients]
        chosen=round_robin(class_ranks,min(top_k,len(nonzero)))
        rule="Balanced one-versus-rest L1 logistic coefficients; round-robin per class; only nonzero dimensions retained"
    elif method=="balanced_trees":
        estimator=ExtraTreesClassifier(n_estimators=192,max_depth=6,min_samples_leaf=3,max_features=.4,class_weight="balanced",n_jobs=2,random_state=seed)
        estimator.fit(z,labels);weights=estimator.feature_importances_
        chosen=np.argsort(-weights,kind="stable")[:top_k].astype(int).tolist()
        rule="ExtraTrees class-balanced impurity importance, fixed depth/seed; correlated features can share importance"
    elif method=="balanced_mrmr":
        pool=np.unique(np.concatenate([r[:128] for r in ranks]))
        correlation=np.nan_to_num(np.corrcoef(z[:,pool],rowvar=False),nan=0.)
        lookup={int(feature):i for i,feature in enumerate(pool)}
        chosen=[]
        while len(chosen)<top_k:
            for row in fisher:
                available=np.array([i for i in pool if int(i) not in chosen])
                if not len(available):break
                redundancy=np.max(np.abs(correlation[np.ix_([lookup[int(i)] for i in available],[lookup[i] for i in chosen])]),axis=1) if chosen else np.zeros(len(available))
                score=row[available]*(1-.85*redundancy)
                chosen.append(int(available[np.argmax(score)]))
                if len(chosen)==top_k:break
        rule="Pair-balanced greedy relevance/redundancy selection: Fisher score times (1-0.85*maximum absolute selected-feature correlation)"
    else:raise ValueError(f"Unknown selector {method}")
    class_means={str(int(g)):means[i,chosen].tolist() for i,g in enumerate(classes)}
    return {"method":method,"indices":chosen,"n_selected":len(chosen),"rule":rule,"selected_weights":weights[chosen].tolist(),
            "standardized_group_means":class_means,"classes":classes.astype(int).tolist(),
            "top_dimensions_by_pair":[{"class_a":int(classes[a]),"class_b":int(classes[b]),
                "indices":ranks[j][:8].astype(int).tolist(),"fisher_scores":fisher[j,ranks[j][:8]].tolist()}
                for j,(a,b) in enumerate(pairs)],"fine_annotation_used":False,"feature_dimension_ids_are_not_biomarkers":True}


def blocked_validation(values,labels,spatial_groups,method,top_k=64):
    """Nested feature selection in each held-out spatial-group split."""
    labels=np.asarray(labels);classes=np.unique(labels);prediction=np.zeros_like(labels)
    rows=[]
    splitter=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=430)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",FutureWarning)
        for fold,(train,test) in enumerate(splitter.split(values,labels,spatial_groups)):
            missing=sorted(set(classes)-set(np.unique(labels[train])))
            if missing:
                rows.append({"fold":fold,"status":"unavailable: training split misses a class","missing_classes":list(map(int,missing)),"n_test":len(test)})
                continue
            selected=select_features(values[train],labels[train],method,top_k)["indices"]
            scaler=StandardScaler().fit(values[train][:,selected])
            classifier=LogisticRegression(C=1.,class_weight="balanced",max_iter=2500,random_state=430)
            classifier.fit(scaler.transform(values[train][:,selected]),labels[train])
            prediction[test]=classifier.predict(scaler.transform(values[test][:,selected]))
            rows.append({"fold":fold,"n_train":len(train),"n_test":len(test),"features":len(selected),"status":"complete",
                         "train_spatial_groups":np.unique(spatial_groups[train]).astype(int).tolist(),
                         "test_spatial_groups":np.unique(spatial_groups[test]).astype(int).tolist()})
    matrix=confusion_matrix(labels,prediction,labels=[*classes,0])
    recall=np.diag(matrix)[:-1]/matrix[:-1].sum(1)
    return {"method":method,"folds":rows,"per_class_recall":dict(zip(map(lambda g:str(int(g)),classes),map(float,recall))),
            "balanced_accuracy":float(recall.mean()),"worst_class_recall":float(recall.min()),
            "complete_heldout_predictions":bool(np.all(prediction>0)),"confusion_matrix_including_unavailable_code_0":matrix.tolist(),
            "class_order":[*classes.astype(int).tolist(),0],"protocol":"4-fold stratified spatial groups of 3x3 coarse tiles; feature selection and scaling refit inside each training fold",
            "limitation":"no spatial buffer; overlapping encoder context can cross folds; one ROI, not patient-level validation"}
