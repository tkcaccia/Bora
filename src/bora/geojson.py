import json
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon, mapping
from skimage import measure


def labels_to_geojson(labels, simplify=0.0, min_area=0.0):
    features = []
    for raw_id in np.unique(labels):
        lid = int(raw_id)
        if lid <= 0: continue
        components = measure.label(labels == raw_id, connectivity=2)
        for cid in range(1, int(components.max())+1):
            region, area = components == cid, int((components == cid).sum())
            if area < min_area: continue
            for contour in measure.find_contours(np.pad(region, 1), .5, fully_connected="high"):
                poly = Polygon([(float(c-1), float(r-1)) for r, c in contour])
                if not poly.is_valid: poly = poly.buffer(0)
                if simplify: poly = poly.simplify(simplify, preserve_topology=True)
                if not poly.is_empty and poly.area >= min_area:
                    features.append({"type":"Feature", "properties":{"label":lid,"value":lid,
                        "classification":f"Cluster {lid}","area_px":area}, "geometry":mapping(poly)})
    return {"type":"FeatureCollection", "features":features}


def write_geojson(path, labels, simplify=0.0, min_area=0.0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(labels_to_geojson(labels, simplify, min_area)))
