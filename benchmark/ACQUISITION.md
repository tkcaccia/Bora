# Acquisition record: 15 October 2026

This record describes downloads and image/mask validation only. **Bora was not run.**
The complete machine-readable provenance is in [acquisition-2026-10-15.json](acquisition-2026-10-15.json).

Data root on Chiamaka: `/mnt/sata_ssd/Bora-benchmark`.

| Data staged | Verified pairs | Source partitions | Status |
|---|---:|---|---|
| bcss | 151 | unsplit: 151 | Downloaded; all pairs inventoried |
| tiger | 2,030 | unsplit: 2030 | Downloaded; all pairs inventoried |
| glas | 165 | testA: 60, testB: 20, train: 85 | Downloaded; all pairs inventoried |
| crag-candidate | 213 | test: 40, train_sup_16: 16, train_unsup_137: 137, val: 20 | Candidate: original provenance and mask fidelity unverified |

PAIP 2019 was not downloaded: provider-approved access is required. Canonical CRAG v2 remains pending provider access. CAMELYON16, SICAPv2 and SegPath are catalogued as optional extensions and were not downloaded.

## Checks performed

- Every acquired BCSS Parquet shard matched its published LFS SHA256. The HF revision is `502d4a3fbc77dbaca6f4664c19e2379ff077d418`.
- The complete TIGER ZIP matched the publisher MD5 `51d65ecc2837a98ff5ceee17fdedd7bd`; a SHA256 was also recorded. ZIP members were extracted with path checks and CRC verification.
- GlaS and CRAG candidate archive hashes were measured after download and pinned for repeat acquisition. These are observed mirror hashes, not original publisher authentication.
- All 2,559 image/mask pairs have aligned dimensions, uint8 RGB images and two-dimensional integer masks; individual file SHA256 values are in the remote JSONL inventories.
- Three acquisition integrity tests passed: checksum rejection, archive traversal rejection, and repeat extraction of a verified archive. These tests do not import or execute Bora.

## Findings relevant to a manuscript

BCSS has 151 pairs and 151 distinct supplied TCGA patient identifiers. Its RGBs are already color-normalized by the mirror. The original 22-value mask encoding is retained; no classes were merged. No evaluation split was assigned.

TIGER contains 151 `tissue-bcss` pairs and 1,879 `tissue-cells` pairs from 195 source slides. There are 1,895 TCGA-derived ROIs and 135 non-TCGA ROIs from 44 non-TCGA slides. The TCGA material overlaps BCSS/NuCLS and must not be presented as independent external validation. The 135 ROIs are not 135 independent patients. Non-TCGA patient mapping was not verified.

GlaS preserves all 85 training, 60 Test A and 20 Test B images. `Grade.csv` identifies 16 patient IDs, of which 11 occur in multiple official image partitions. The inventory preserves the official partitions and records this overlap; it does not manufacture a patient-independent split.

The CRAG candidate contains 213 pairs partitioned by its uploader into 16 supervised-training, 137 unsupervised-training, 20 validation and 40 test images. These are mirror-specific partitions. Masks contain integer labels, but their identity with the original CRAG v2 masks has not been established. Candidate files and examples are kept explicitly separate from canonical CRAG.

No stain transformation, spatial resizing, coarse-label generation, model inference or accuracy calculation was performed by this workflow. The dataset inventories inspect pixel values and geometry only.

## Archive identities

| Archive | Bytes | SHA256 |
|---|---:|---|
| `raw/tiger/roi-level-annotations.zip` | 2,610,151,375 | `94bf1a00a61b8d264a6d8d9f213000617766ce65823ab33af86498041bf866dd` |
| `raw/glas/glas-v1.zip` | 180,682,993 | `2f9ca81961832c497c32110a63f96e007f996a946a014cc58768e9dad566887e` |
| `raw/crag-candidate/crag-candidate-v1.zip` | 1,067,055,555 | `b06ecdabe95d6e3b9062dff287126650fc695bb4b754b8ac6dc05fec78989c47` |

## Reuse and local examples

The `examples.py` script copies three original RGB images and their raw masks per available dataset. TIGER examples cover the BCSS-derived subset and non-TCGA tissue ROIs; each is at least 1000 pixels on both sides. A previously selected cell-scale crop was replaced together with its mask. Example files are a convenience sample; they are not an evaluation cohort. Local copies are saved under `benchmark/examples/` and excluded from Git. `examples.json` retains their source IDs and hashes. PAIP examples remain unavailable. Resolution evidence, annotation legends and the replacement sample identity are documented in [ANNOTATIONS.md](ANNOTATIONS.md).

The isolated acquisition environment, inventories and logs are retained under the remote data root. An environment freeze is at `metadata/environment.txt`. Download chunks and the interrupted sequential TIGER partial are retained for recovery; these account for additional disk use. No cloud resource was provisioned and no scheduler job was submitted. At the completion audit the data root occupied approximately 21 GiB, including archives, extracted files, examples, environment and recovery chunks.

Python and acquisition library versions:

- python: `3.10.12`
- huggingface_hub: `2.2.0`
- pyarrow: `25.0.1`
- Pillow: `12.3.0`
- numpy: `2.2.6`
