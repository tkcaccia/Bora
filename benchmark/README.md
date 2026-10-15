# Histopathology benchmark data for Bora

This folder contains acquisition and inventory code, source locks, and dataset
notes for a future Bora manuscript. **Bora is not run by any of these commands.**
No benchmark scores, model predictions, or derived coarse labels are produced.

Read [DATASETS.md](DATASETS.md) for annotation definitions, source citations,
access conditions, overlap, and limitations. [ACQUISITION.md](ACQUISITION.md)
records the actual acquisition on 15 October 2026.
Read [ANNOTATIONS.md](ANNOTATIONS.md) for microscopy sampling, tumor/stroma
label maps and per-image annotation metadata. Source-qualified spacing is
stored in [resolution.json](resolution.json).

## Storage

The Chiamaka data root is `/mnt/sata_ssd/Bora-benchmark`:

```text
Bora-benchmark/
  code/             # copy of Bora source and this benchmark folder
  .venv/            # isolated Python environment
  raw/              # downloaded original archives and Parquet files
  prepared/         # original BCSS PNG bytes and JSONL pair inventories
  metadata/         # acquisition hashes, inventory summaries, logs, environment
  examples/         # three original image/mask pairs per available dataset
  results/          # reserved; no Bora evaluation has been run
```

BCSS is distributed as Parquet; inventory preparation extracts the embedded
original PNG bytes. Other datasets retain their original extracted image and
mask bytes. Images are not resized, stain-normalized, or passed through Bora.
The BCSS mirror already contains normalized images; that source preprocessing
is recorded separately. Preserve the downloaded archives as provenance.

Dataset archives, generated inventories and local example images must remain
outside Git. The local examples directory `benchmark/examples/` is ignored.

## Reproduce the acquisition

Run from the Bora repository root with Python 3.10 or newer:

```bash
python3 -m venv /mnt/sata_ssd/Bora-benchmark/.venv
source /mnt/sata_ssd/Bora-benchmark/.venv/bin/activate
python -m pip install -r benchmark/requirements.txt numpy Pillow pytest

python -m benchmark.download bcss --plan
python -m benchmark.download bcss
python -m benchmark.download tiger --extract --workers 8
python -m benchmark.download glas --extract

python -m benchmark.prepare bcss
python -m benchmark.prepare tiger
python -m benchmark.prepare glas
python -m benchmark.examples --destination /mnt/sata_ssd/Bora-benchmark/examples --count 3
python -m benchmark.describe --examples /mnt/sata_ssd/Bora-benchmark/examples
python -m pytest -q benchmark/tests/test_acquisition.py
```

The Python environment does not need Bora installed for these commands. The
Hugging Face CLI downloads the exact BCSS revision in `sources.lock.json`, and
each Parquet file is checked against its published LFS SHA256. TIGER is checked
against Zenodo's MD5 and a local SHA256 is recorded. GlaS is checked against an
observed SHA256 of the pinned Kaggle version. Downloads preserve at least
20 GiB of free space by default. Unknown-size mirrors are checked while
streaming. Large HTTPS files support up to eight checked byte-range requests;
completed chunks and an interrupted sequential transfer can be reused. Partial
downloads and chunk files are retained for recovery, and count toward disk use.

ZIP paths, symlinks, duplicate names, extraction collisions and free space are
checked before extraction. A completed extraction is identified by its archive
hash. An interrupted extraction is not overwritten: inspect it and use a new
data root if necessary. Inventory commands verify every image/mask shape and
record each input's SHA256. BCSS, TIGER and GlaS pair counts are checked against
the pinned releases. Existing inventories with different content are rejected.
No evaluation split is assigned by this acquisition workflow.
The description command writes a resolution/annotation CSV for every acquired
image and creates per-example metadata plus an interactive local mask viewer.
It does not alter the original acquisition inventories or input image bytes.

## CRAG and PAIP access

The official CRAG page currently redirects to Warwick sign-in. The public
candidate mirror can be staged for inspection, separately from canonical CRAG:

```bash
python -m benchmark.download crag-candidate --extract
python -m benchmark.prepare crag-candidate
```

Its uploader gives no description or license, and its original annotation
provenance is unverified. Keep this candidate out of manuscript analyses until
it is reconciled with the provider's CRAG v2 release. The acquisition report
records its observed structure rather than assuming that masks are unchanged.

For provider-approved CRAG and PAIP archives:

```bash
python -m benchmark.download crag --plan
python -m benchmark.download crag --local-archive /path/to/CRAG_v2.zip --extract
python -m benchmark.prepare crag

python -m benchmark.download paip2019 --plan
python -m benchmark.download paip2019 --local-archive /path/to/approved-training.zip --extract
```

PAIP requires the user's approved data-use agreement and provider delivery.
This code does not register accounts, sign agreements or handle access secrets.
PAIP archives can be inventoried as files, but WSI/annotation window preparation
requires the acquired files and verified spatial metadata; no speculative PAIP
pixel adapter is provided. Do not replace its official binary TIF masks by a
fresh rasterization of the XML contours.

The local-archive option accepts one ZIP. For multipart deliveries, preserve
the original parts and import each into a separate root. Dataset access
conditions and acquisition details are in the linked report.
