import gzip
import zipfile

import pytest

from train.covertype_data import DATA_SHA256, load_covertype, original_csv_bytes
from train.covertype_experiment import configurations, multiclass_metrics
from train.run import Config


def test_covertype_archive_accepts_original_packaging(tmp_path):
    content = b'1,2,3\n4,5,6\n'
    archive_path = tmp_path / 'covertype.zip'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('nested/covtype.data.gz', gzip.compress(content))
    assert original_csv_bytes(archive_path) == content
    plain_path = tmp_path / 'covtype.data'
    plain_path.write_bytes(content)
    assert original_csv_bytes(plain_path) == content
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('other.txt', content)
    with pytest.raises(ValueError, match='Expected exactly one'):
        original_csv_bytes(archive_path)


def test_covertype_protocol_pairs_identical_training_configs():
    entries = configurations([0, 1, 2])
    assert len(entries) == 9
    assert len({entry['directory'] for entry in entries}) == 9
    for seed in (0, 1, 2):
        group = [entry for entry in entries if entry['seed'] == seed]
        assert {entry['method'] for entry in group} == {'dense', 'magnitude', 'hessian'}
        for entry in group:
            Config(**entry['config']).validate()
        first = group[0]['config']
        for entry in group[1:]:
            assert {key: value for key, value in entry['config'].items()
                    if key not in ('method', 'sparsity')} == {
                        key: value for key, value in first.items()
                        if key not in ('method', 'sparsity')}


def test_multiclass_metrics_include_classes_never_predicted():
    import numpy as np
    labels = np.arange(7)
    predictions = np.array([0, 1, 1, 3, 4, 4, 6])
    logits = np.eye(7)[predictions]
    result = multiclass_metrics(logits, labels)
    assert sum(result['predicted_class_counts']) == 7
    assert result['per_class_recall'] == [1, 1, 0, 1, 1, 0, 1]
    assert result['balanced_accuracy'] == 5/7
    assert result['confusion_matrix'][2][1] == 1
    assert result['confusion_matrix'][5][4] == 1


def test_bundled_covertype_split_and_training_scaling():
    import numpy as np
    data = load_covertype()
    assert data['metadata']['sha256'] == DATA_SHA256
    assert data['metadata']['split_sizes'] == {
        'train': 348605, 'val': 116202, 'test': 116205}
    assert sum(len(data[name][1]) for name in ('train', 'val', 'test')) == 581012
    x = data['train'][0]
    np.testing.assert_allclose(x[:, :10].mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(x[:, :10].std(axis=0), 1, atol=1e-10)
    assert np.isin(x[:, 10:], [0, 1]).all()
