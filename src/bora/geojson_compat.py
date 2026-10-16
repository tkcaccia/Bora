"""Add annotation metadata expected by wsiTools to categorical GeoJSON."""
from __future__ import annotations

import json
from pathlib import Path


def enrich_wsitools_geojson(vector_path: Path, metrics_path: Path,
                           width: int, height: int) -> dict:
    """Preserve one feature per class and attach source class names and colors."""
    vector_path = Path(vector_path)
    metrics = json.loads(Path(metrics_path).read_text())
    classes = {int(item["value"]): item for item in metrics["classes"]
               if int(item["value"]) > 0}
    data = json.loads(vector_path.read_text())
    seen = set()
    for feature in data["features"]:
        value = int(feature["properties"]["value"])
        if value not in classes:
            raise ValueError(f"GeoJSON class {value} is absent from tiling-mask metadata")
        item = classes[value]
        name = str(item["name"])
        red, green, blue = map(int, item["color"])
        color = f"#{red:02X}{green:02X}{blue:02X}"
        props = feature["properties"]
        props.update({
            "objectType": "annotation",
            "name": name,
            "label": name,
            "class": name,
            "classification": {"name": name, "color": color},
            "isLocked": False,
            "visible": True,
            "coordinate_space": "original_image_level0_pixel_edges",
            "source_type": "annotation",
            "kind": "tissue",
            "tissue_annotation": True,
            "wsiTools": {
                "classPreset": name,
                "export": True,
                "exportRule": "include",
                "coordinate_space": "level0_slide_pixels",
                "coordinateSpace": "level0_slide_pixels",
                "slide_width": int(width),
                "slide_height": int(height),
                "display_transform_applied": False,
            },
        })
        seen.add(value)
    expected = set(classes)
    if seen != expected:
        raise ValueError(f"GeoJSON annotations missing classes: {sorted(expected-seen)}")
    temporary = vector_path.with_suffix(vector_path.suffix + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(vector_path)
    return {"classes": sorted(seen), "features": len(data["features"])}
