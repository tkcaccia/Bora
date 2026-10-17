"""Active benchmark scope; rejected implementations are absent."""
STRATEGIES = {
 'B02': ('Whole-field appearance classification', 'Class-conditional Gaussian appearance learned from original coarse interiors relabels the entire field; sparse tile-center anchors.'),
 'B03': ('Multiscale posterior agreement', 'Fit class appearance independently after 1, 4 and 12 µm averaging; combine posteriors geometrically over the whole field.'),
 'B05': ('Seeded watershed', 'Flood the converted-image gradient from sparse original tile-center seeds; all pixels participate.'),
 'B08': ('Finite-range CRF', 'Mean-field posterior inference with contrast-weighted messages at three physical neighborhood radii.'),
 'B09': ('Geodesic competition', 'Each coarse class expands along minimum cumulative appearance/edge travel cost; competing geodesic distances select the label.'),
 'B10': ('Convex total-variation partition', 'Primal-dual optimization of a simplex-valued class field with appearance fidelity and isotropic TV regularity.'),
 'B12': ('Hierarchical region merging', 'Agglomerate adjacent image regions by adaptive mean-color distance while preventing conflicting coarse-anchor mergers.'),
 'B19': ('Retained-solver consensus', 'Equal votes from B02 appearance, B05 watershed, B09 geodesic and B10 TV; appearance resolves ties. Revised membership after rejected solvers were removed.'),
}
CONVERSIONS = ('09','10','15','19','ensemble')
REJECTED_STRATEGIES = ('B04','B20','B16','B17','B14','B13','B01','B06','B07','B11','B15','B18')
REJECTED_CONVERSIONS = (4,5,8)
CONSENSUS = ('B02','B05','B09','B10')
PARAMETERS = dict(physical_target_um=1.,random_seed=430,training_core_um=12.6,
 region_spacing_um=15.,region_limit=4096,crf_radii_um=[1,4,12],crf_iterations=10,
 crf_weight=1.2,tv_iterations=45,tv_weight=.18,merge_LAB_threshold=18.)
