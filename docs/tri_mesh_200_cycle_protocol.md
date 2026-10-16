# Regenerated-mask, triangular-mesh benchmark protocol

## Benchmark version

The four input images and current GeoJSON annotations are under
`/mnt/sata_ssd/benchmarks/Bora/input`. The annotations were copied and hashed
under `/mnt/sata_ssd/benchmarks/Bora/benchmark-regenerated-2026-10-15` before
evaluation. New 56, 90, and 216-pixel masks were generated into
`/mnt/sata_ssd/benchmarks/Bora/output-tilling-mask-regenerated-2026-10-15`.
The original `output-tilling-mask` directory remains unchanged while the
preceding cycle-50 run finishes.

The regenerated annotations differ materially from the older mask set:
lymph node has two foreground classes instead of five, and brain has five
instead of four. Thus results from the old cycle-12/50 benchmark are **not**
numerically comparable with results using the regenerated mask set.

## Method

The new C++ stage samples class occupancy and, optionally, local RGB class
means at the vertices of a regular triangular mesh. It interpolates class
support within each triangle and assigns the winning class inside a bounded
band around **internal tissue-to-tissue borders**. Foreground/background
membership and tissue interiors outside the band are fixed. Tile-size-adaptive
mesh spacing, support radius, and editing band allow fine masks to retain
detail while coarser masks receive broader smoothing. An optional C++ majority
pass rounds mesh corners **after** triangulation. Where local class colors are
indistinguishable, occupancy support expands by about 30%; where colors
separate clearly, it stays narrower so the contour can follow image evidence.
The stage uses OpenMP;
Python dispatches independent WSI blocks to multiple processes with aligned
mesh origins and bounded memory.

Build with `cmake -S cpp -B cpp/build && cmake --build cpp/build -j 4`.
The generated shared library is ignored by Git. Source code is in
`cpp/tri_mesh_refine.cpp`, with the streaming wrapper in `src/bora/tri_mesh.py`.

## Evaluation

Each numbered cycle from 51 through 250 will export four whole-slide GeoJSONs
at each of the three tile sizes. GeoJSONs are rasterized at native resolution
against the matching regenerated pixel-level truth. Primary scoring is within
64 pixels of a ground-truth or original tile **internal** interface, excluding
ground-truth background. We record macro IoU, accuracy, 8/16/32-pixel boundary
F1, average symmetric surface distance, robust 95th-percentile Hausdorff
distance, false tile-edge pixels, class retention, and timing. Pancreas has
only one foreground class, so internal-border metrics are not applicable
there; its output and runtime are still checked. Internal edge pixels outside
both the truth and tile 64-pixel bands are recorded separately as a spurious
contour diagnostic.

The regenerated reference pipeline starts with native watershed, applies the
existing gentle contour pass (sigma 12, band 28, small holes up to 1500
pixels), removes enclosed islands up to 1500 pixels, then applies the accepted
strong sigma-80/band-216 smoothing. The triangular method replaces only that
last strong pass. This preserves the earlier foreground and island cleanup
stages while making the speed and accuracy comparison meaningful.

A candidate becomes the new incumbent only when internal-border macro IoU and
accuracy do not decline for any tissue and tile size, false tile-edge pixels
and off-band internal-edge pixels do not increase, all ground-truth classes remain represented, and at least
one measured quality or speed value improves. The narrow-tolerance contour
metrics remain diagnostic even when this gate accepts a candidate. The four
slides are used for both tuning and evaluation, so final scores are
development-set results, not evidence of generalization to unseen slides.

## Initial checks

The C++ kernel passed synthetic boundary, class-retention, color-guidance,
and WSI block-equivalence tests. On one 2,048-pixel brain crop, its contour
step ran in 0.056 s versus 1.614 s for the current Python Gaussian step. This
is a **kernel-only** comparison; end-to-end WSI speed is measured separately.

An exploratory probe of 36 crops from the **regenerated raw tile masks**
showed improved mean border-band accuracy with tile-size-adaptive settings
for lymph node, tonsil, and brain. It does not establish superiority over the
current accepted Bora output. Full-resolution GeoJSON evaluation determines
acceptance.
The first contrast-adaptive, post-smoothed candidate was rechecked on four
regenerated crops for each of the nine multiclass tissue/size combinations;
all nine mean border-band accuracy changes versus the raw tile mask were
positive. The exact settings and per-crop changes are stored in
`tri_mesh_adaptive_crop_probe.json` in the versioned benchmark directory.

On four additional 216-pixel crops per tissue, contrast-gated local RGB
guidance improved mean border-band accuracy for lymph node and tonsil as its
weight increased to 0.4, while it had no measurable effect on brain. Earlier
unconditional RGB weighting reduced brain accuracy; the C++ kernel therefore
uses local between-class color separation to suppress unreliable color cues.

Before the regenerated full benchmark started, a **separate pilot on the old
216-pixel truth** replaced the cycle-12 Gaussian stage with the C++ triangular
stage from the same cycle-07 source. All four GeoJSONs were produced. Relative
to cycle 12, the three multiclass tissues improved internal-border macro IoU,
accuracy, 8-pixel F1, false tile-edge count, and measured experimental pipeline
time. Pancreas was copied exactly because it has no internal class boundary.
The quantitative pilot is in `tri-mesh-adaptive-old-216/pilot_comparison.json`.
Across the three multiclass slides, mean internal-border macro IoU rose by
0.008227, mean border accuracy by 0.007821, and mean 8-pixel boundary F1 by
0.018518; 11,492 false tile-edge pixels were removed. Summed experimental
pipeline time was 1.92 times faster than cycle 12. All old truth classes were
retained. HD95 and other diagnostics remain separately reported.
This pilot does **not** establish performance on the regenerated annotations
or the 56- and 90-pixel masks.

## Operational state

`scripts/watch_cycles_to_250.py` waits for the preceding cycle-50 search,
prepares the regenerated baselines, and starts the resumable 200-cycle runner.
The runner writes state to `cycles_51_to_250.json` in the versioned benchmark
directory. Each cycle has its own numbered output and log. A separate hourly
Codex heartbeat checks progress and handles failures; it should pause after
cycle 250 and the final report are verified.
