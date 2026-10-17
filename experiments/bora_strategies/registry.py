"""Active benchmark scope; rejected implementations are absent."""
STRATEGIES = {
 'B02': ('Whole-field appearance classification', 'Class-conditional Gaussian appearance learned from original coarse interiors relabels the entire field; sparse tile-center anchors.'),
 'B03': ('Multiscale posterior agreement', 'Fit class appearance independently after 1, 4 and 12 µm averaging; combine posteriors geometrically over the whole field.'),
 'B08': ('Finite-range CRF', 'Mean-field posterior inference with contrast-weighted messages at three physical neighborhood radii.'),
 'B19': ('Retained-solver consensus', 'Equal votes from retained B02/B03/B08; appearance resolves ties. Revision 3 excludes newly rejected solvers.'),
}
CONVERSIONS = ('09','10','15','19','ensemble')
REJECTED_STRATEGIES = ('B04','B20','B16','B17','B14','B13','B01','B06','B07','B11','B15','B18','B05','B09','B10','B12')
REJECTED_CONVERSIONS = (4,5,8)
CONSENSUS = ('B02','B03','B08')
PARAMETERS = dict(physical_target_um=1.,random_seed=430,training_core_um=12.6,
 region_spacing_um=15.,region_limit=4096,crf_radii_um=[1,4,12],crf_iterations=10,
 crf_weight=1.2)
