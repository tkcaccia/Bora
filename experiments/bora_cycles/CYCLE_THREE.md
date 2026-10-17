# Cycle three: models derived from the full cycle-two review

Two of twenty requested cycles are fully reviewed and selected. Cycle three
keeps **D10, C17, C01, D08 and C15** unchanged, with exact native-mask parity
against cycle two required on every image. The fifteen F-series children
are experimental interventions, not claims of methodological novelty.

The previous review found real white tears, bright labelled lumina and fat,
missed thin vessels, false colour rims and weak tile-likelihood fits that had
not converged. No method can recover an absent biological class identity from
RGB novelty alone. Those limitations motivate these tests, not a promise of
improvement. No rejected solver is restored under a new ID.

| ID | Parent | Intervention | Main risk |
|---|---|---|---|
| F01 | D10 | Integrate local affine prototype uncertainty in the predictive covariance and its normalization. | Unsupported classes become too tolerant, or rare classes pay an uncertainty penalty. |
| F02 | D10 | Exclude the query's entire coarse tile from its local affine sufficient statistics. | A one-tile class loses its only local evidence; the global prior still uses all coarse tissue. |
| F03 | D10 | A glass state for whole elongated, narrow white components, measured in physical coordinates. | Elongated genuine lumina can be deleted; broad empty cavities remain. |
| F04 | C17 | Invert a spatial Gaussian forward-blur model of palette mixtures. | An incorrect point-spread assumption amplifies noise or creates ringing. |
| F05 | C17 | Estimate a single identity-shrunk affine observation map before palette inversion. | Mixed coarse labels bias the inferred palette geometry. |
| F06 | C17 | Piecewise affine class coordinates on an intrinsic palette Delaunay tessellation. | Tessellation chooses the wrong mixture branch; outside-hull extrapolation is abrupt. |
| F07 | C01 | Convex tile-label cross entropy of mean pixel logits, solved by analytic damped Newton steps. | Tile means discard the structure that determines a true count winner. |
| F08 | C01 | Log-mean-exp representative-pixel evidence for each class within a tile. | A spurious colour witness explains the tile and then expands elsewhere. |
| F09 | C01 | Analytic ridge regression on kernel mean embeddings of tile colour distributions. | A tile-distribution predictor may not transfer to individual pixels. |
| F10 | D08 | Estimate nuisance covariance from adjacent within-tile colour increments. | Spatially correlated noise or real microscopic boundaries violate the noise model. |
| F11 | D08 | Separate latent between-tile appearance variance from within-tile sampling noise. | Few bags produce unstable class-specific covariance and engulf small tissue. |
| F12 | D08 | Fit joint squared all-rival margins in within-tile noise-whitened colour coordinates. | Coarse tile means force the wrong separating plane. |
| F13 | C15 | Distributional energy score from expected RGB distances, evaluated by FFT potentials. | Broad distributions obscure small distinctive modes. |
| F14 | C15 | Signed geometric depth in learned class colour-support hulls. | Convex support bridges holes and is enlarged by contaminated training colours. |
| F15 | C15 | Compare held-out class-specific density ranks rather than raw density heights. | Broad classes look typical for many colours; weak spatial samples give no coverage guarantee. |

The registry contains each equation/rule, lineage and failure hypothesis.
The implementation is `cycle_three.py`. F-series IDs avoid confusion with
the earlier E-series development experiments.

F03 uses original H&E minima >=235 and channel range <=18 to propose white
components, then requires area >=20 um2, inertia length >=16 um, axis ratio
>=4, and area/length <=10 um. It does not use annotation or border connection
to identify glass. All other children retain the original background support.
Every candidate is scored on all original reference-positive pixels,
including annotated pixels F03 deletes. Primary masks assign known coarse
tissue IDs or background zero. Unknown-colour proposals have no invented
tissue name and are reported separately from primary segmentation.

Most analysis is at 1 um/pixel; actual per-axis spacing and native projection
are recorded. Spacing does not establish microscope optical resolution.
All five original RGB conversions remain byte-identical; the primary cycle
comparison uses fixed equal 9+19 on all 149 ROIs. No Virchow2 inference or
fine-reference training is introduced.

The C++ OpenMP hull kernel streams facet tests to avoid allocating a large
pixel-by-facet matrix. Its NumPy and one-versus-two-thread parity tests are
engineering verification, not additional candidate algorithms. Existing
compiled palette and linear kernels are reused where appropriate. Three
low-priority image processes use at most two numerical/OpenMP threads each.
Timing includes each method's preparation, inference and native projection,
and excludes shared I/O, scoring, compression and rendering.

New mathematical tests check tile subtraction against independent weighted
normal equations; a slit versus round lumen and physical-scale counterexample;
the blur objective against a separate constrained optimizer; tetrahedral
barycentric identities; likelihood/hinge gradients and Newton Hessian; FFT
energy versus all-pairs distances; C++ facets versus NumPy; and equality of
uncalibrated F15 density with its C15 parent.

A four-group pilot is an engineering and visual check, not a completed cycle
or a selection cohort. The full run freezes all predictions before any fine
reference is opened. Only actual review of all 149 all-method sheets, scores,
time and native failures permits the next five-survivor selection. Repeated
selection on these images is development, not independent validation.

The first four-group pilot is preserved as cycle-03-pilot-v1. It exposed
zero-iteration failures in two F12 maximum-rival fits because the loss was
nondifferentiable at tied rivals. Before the full cycle, F12 was revised to
the mean of squared all-rival hinges, which has a continuous gradient at
ties. An explicit tied-initialization gradient check covers this failure.
The corrected source and a fresh pilot are version two; v1 is not a completed
cycle and its failed-fit masks are not silently substituted into the full run.
