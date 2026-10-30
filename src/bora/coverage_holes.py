"""Isolated correction for partially occupied enclosed background holes.

All areas are expressed in the coordinate system of the supplied GeoJSON.
The union of predicted tissue, rather than any one class's rings, defines the
background holes. No predicted tissue is overwritten and exterior background
is never painted.
"""
import math


def fill_small_vector_holes(data, minimum_pixels=20):
    from shapely.geometry import shape, Polygon, mapping
    from shapely.ops import unary_union
    from shapely.strtree import STRtree
    from shapely.affinity import scale
    if not math.isfinite(float(minimum_pixels)) or minimum_pixels <= 0:
        raise ValueError('Positive finite vector hole area threshold required')
    geometries = [shape(feature['geometry']) for feature in data['features']]
    if any(not g.is_valid for g in geometries):
        raise ValueError('Invalid geometry before enclosed-background cleanup')
    coverage = unary_union(geometries)
    parts = [coverage] if coverage.geom_type == 'Polygon' else list(getattr(coverage, 'geoms', []))
    holes = [Polygon(ring) for part in parts if part.geom_type == 'Polygon'
             for ring in part.interiors if Polygon(ring).area < minimum_pixels]
    # Stable order and immutable original geometries make contact ties reproducible.
    holes.sort(key=lambda p: (-p.area, p.wkb_hex))
    originals = tuple(geometries)
    tree = STRtree(originals)
    spacing = data.get('native_mpp_xy', [1., 1.])
    filled = 0
    filled_area = 0.
    for hole in holes:
        candidates = []
        for index in tree.query(hole):
            original = originals[int(index)]
            contact = original.boundary.intersection(hole.boundary)
            contact_length = scale(contact, xfact=spacing[0], yfact=spacing[1], origin=(0, 0)).length
            if contact_length > 0:
                candidates.append((contact_length, original.area, -int(index), int(index)))
        if not candidates:
            raise ValueError('Enclosed background has no contacting predicted tissue')
        winner = max(candidates)[-1]
        # Holes of the union contain no tissue from any class. Only that empty
        # component is assigned; occupied portions of class rings stay intact.
        geometries[winner] = geometries[winner].union(hole)
        if not geometries[winner].is_valid:
            raise ValueError('Enclosed-background filling produced invalid geometry')
        filled += 1
        filled_area += hole.area
    for feature, geometry in zip(data['features'], geometries):
        feature['geometry'] = mapping(geometry)
        if 'area_pixels' in feature['properties']:
            feature['properties']['area_pixels'] = float(geometry.area)
        if 'area_um2' in feature['properties'] and 'native_mpp_xy' in data:
            feature['properties']['area_um2'] = float(geometry.area * math.prod(spacing))
    data['small_vector_hole_cleanup'] = dict(
        minimum_hole_area_pixels=minimum_pixels, filled_holes=filled, filled_area_pixels=filled_area,
        strictly_less_than_threshold=True, hole_definition='enclosed unoccupied components of all-class tissue union',
        coverage_exterior_rings_unchanged=True, other_predicted_tissue_not_overwritten=True,
        assignment='longest physical shared boundary; ties prefer larger neighboring class then lower feature index')
    return data
