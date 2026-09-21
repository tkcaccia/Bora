"""Lossless pyramidal OME-TIFF conversion for categorical masks."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile


def pyramidize_mask(path, compression="LZW", workers=8):
    """Replace a flat TIFF with a tiled multiresolution OME-TIFF atomically."""
    source = Path(path).resolve()
    b2r = shutil.which("bioformats2raw")
    r2o = shutil.which("raw2ometiff")
    if not b2r or not r2o:
        raise RuntimeError("Pyramidal output requires bioformats2raw and raw2ometiff on PATH")
    work = Path(tempfile.mkdtemp(prefix=f".{source.name}.pyramid-", dir=source.parent))
    rawdir, converted = work / "pixels.raw", work / source.name
    env = os.environ.copy()
    if env.get("JAVA_HOME") and not Path(env["JAVA_HOME"]).is_dir():
        env.pop("JAVA_HOME", None)
    try:
        subprocess.run([b2r, "--log-level=OFF", "--downsample-type", "SIMPLE",
                        str(source), str(rawdir)], check=True, env=env)
        subprocess.run([r2o, f"--compression={compression}",
                        f"--max_workers={workers}", str(rawdir), str(converted)],
                       check=True, env=env)
        os.replace(converted, source)
    finally:
        shutil.rmtree(work, ignore_errors=True)
