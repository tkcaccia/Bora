"""Bora-AF refinement and native fastPLS Neighbor context identification."""
__version__ = "0.3.1"


def segment_full_slide(manifest, output):
    """Predict and vectorize a supervised Bora whole slide on CPU."""
    from .supervised_wsi import segment_full_slide as run
    return run(manifest, output)
