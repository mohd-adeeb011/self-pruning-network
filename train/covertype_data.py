"""Load the original UCI Covertype classification data without extra packages."""
from functools import lru_cache
import gzip
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / 'data'
SOURCE = 'https://archive.ics.uci.edu/dataset/31/covertype'
CLASS_COUNTS = [211840, 283301, 35754, 2747, 9493, 17367, 20510]
DATA_SHA256 = '0a9371cef7c964b5475d6053cc3e0894a5aa6f65ad1ed3ecb01c45aa96217945'


def original_csv_bytes(path):
    """Accept UCI's .zip archive, its .data.gz member, or plain .data file."""
    content = Path(path).read_bytes()
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            names = [name for name in archive.namelist()
                     if Path(name).name in ('covtype.data.gz', 'covtype.data')]
            if len(names) != 1:
                raise ValueError('Expected exactly one covtype.data or covtype.data.gz member.')
            name = names[0]
            content = archive.read(name)
    else:
        name = Path(path).name
    if name.endswith('.gz'):
        content = gzip.decompress(content)
    return content


@lru_cache(maxsize=2)
def load_covertype(seed=2026):
    candidates = [DATA_DIRECTORY / name for name in
                  ('covertype.zip', 'covtype.data.gz', 'covtype.data')]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            'Download the official UCI archive to data/covertype.zip. '
            'See data/README.md for the command.')
    content = original_csv_bytes(path)
    digest = hashlib.sha256(content).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError('UCI Covertype data checksum mismatch.')
    raw = np.loadtxt(io.BytesIO(content), delimiter=',', dtype=np.float64)
    if raw.shape != (581012, 55) or not np.isfinite(raw).all():
        raise ValueError(f'Unexpected UCI Covertype data: shape={raw.shape}.')
    labels = raw[:, -1]
    if not np.array_equal(labels, labels.astype(np.int64)):
        raise ValueError('Expected integer cover type labels.')
    y = labels.astype(np.int64) - 1
    if np.bincount(y, minlength=7).tolist() != CLASS_COUNTS:
        raise ValueError('Unexpected Covertype class counts.')
    x = raw[:, :54]
    rng = np.random.default_rng(seed)
    buckets = [[], [], []]
    for label in range(7):
        subset = rng.permutation(np.flatnonzero(y == label))
        boundaries = [int(.6 * len(subset)), int(.8 * len(subset))]
        for bucket, piece in zip(buckets, np.split(subset, boundaries)):
            bucket.append(piece)
    indices = [rng.permutation(np.concatenate(bucket)) for bucket in buckets]
    mean = x[indices[0], :10].mean(axis=0)
    scale = np.maximum(x[indices[0], :10].std(axis=0), 1e-12)
    # The remaining 44 columns are already 0/1 indicators.
    normalization = (np.r_[mean, np.zeros(44)], np.r_[scale, np.ones(44)])
    data = {}
    for name, idx in zip(('train', 'val', 'test'), indices):
        values = x[idx].copy()
        values[:, :10] = (values[:, :10] - mean) / scale
        data[name] = (values, y[idx].copy())
    data['normalization'] = normalization
    data['metadata'] = dict(
        name='UCI Covertype', source=SOURCE, sha256=digest,
        rows=len(raw), features=54, classes=7,
        split_policy='stratified random 60/20/20; split seed recorded in config',
        split_sizes={name: len(data[name][1]) for name in ('train', 'val', 'test')},
        split_class_counts={name: np.bincount(data[name][1], minlength=7).tolist()
                            for name in ('train', 'val', 'test')},
        preprocessing='First 10 numeric columns standardized on training rows only; 44 binary indicators unchanged.',
    )
    return data
