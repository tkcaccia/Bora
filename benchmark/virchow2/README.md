# Virchow2: twenty experimental ideas for Bora

This benchmark tests twenty distinct interventions on one expert-annotated
TIGER breast tissue ROI. The image is 1224 × 1166 pixels at provider-reported
approximately 0.5 µm/px. It is **an ROI, not a full WSI**; objective magnification
and optical resolving power are unknown.

- [Experimental protocol and reproduction commands](../../experiments/virchow2_inverse/README.md)
- [Experiment source code](../../experiments/virchow2_inverse/)
- Results report: `REPORT.md` in this directory after execution.

The source image and coarse mask come from the downloaded benchmark. No
GigaTIME prediction is used. Fine reference masks are loaded for scoring only
after all twenty outputs are frozen. The coarse labels themselves were derived
from expert annotations, so the task is controlled boundary reconstruction.

On Chiamaka, complete numerical outputs are saved under:

`/mnt/sata_ssd/Bora-benchmark/results/virchow2-inverse-2026-10-16/`

The Mac gallery copy is saved under:

`/Users/stefano/Documents/ChatGPT/Bora/benchmark/results/virchow2-inverse-2026-10-16/`

Open `gallery.html` to compare the original image, expert reference, coarse
tiling mask, each experimental image and each Bora result. Overlays have
adjustable transparency. TIFF masks, per-class scores, model/input hashes,
physical sampling, candidate object maps and donor provenance remain with the
full run. Models, benchmark images and generated outputs are excluded from Git.

Dataset: [TIGER](https://tiger.grand-challenge.org/Data/).
Model: [Virchow2](https://huggingface.co/paige-ai/Virchow2).
