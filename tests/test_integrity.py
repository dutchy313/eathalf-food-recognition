"""Checks for substantive data leakage and metric-order failure modes."""
import numpy as np
import pytest
from PIL import Image

from eathalf import data
from eathalf.workflow import metrics


def image(path, value):
    Image.new('RGB', (8, 8), (value, value // 2, 0)).save(path)


def fixture_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'CLASSES', ('alpha', 'beta'))
    root = tmp_path / 'images'
    for label, start in [('alpha', 0), ('beta', 100)]:
        (root / label).mkdir(parents=True)
        for index in range(20):
            image(root / label / f'{index}.png', start + index)
    return root


def test_disjoint_reproducible_split_and_duplicate_exclusion(tmp_path, monkeypatch):
    root = fixture_data(tmp_path, monkeypatch)
    (root / 'alpha' / 'duplicate.png').write_bytes((root / 'alpha' / '0.png').read_bytes())
    (root / 'alpha' / 'broken.jpg').write_bytes(b'not an image')
    run_a, run_b = tmp_path / 'a', tmp_path / 'b'
    data.prepare(root, run_a)
    data.prepare(root, run_b)
    groups = []
    for split in ('train', 'val', 'test'):
        assert (run_a / f'{split}.csv').read_bytes() == (run_b / f'{split}.csv').read_bytes()
        rows = data.read_manifest(run_a / f'{split}.csv')
        groups.append({r['sha256'] for r in rows})
        assert {r['label'] for r in rows} == {'alpha', 'beta'}
    assert sum(map(len, groups)) == 40
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert len(data.read_manifest(run_a / 'removed_duplicates.csv')) == 1
    assert len(data.read_manifest(run_a / 'invalid_images.csv')) == 1
    with pytest.raises(ValueError, match='Output exists'):
        data.prepare(root, run_a)


def test_conflicting_duplicate_labels_stop_preparation(tmp_path, monkeypatch):
    root = fixture_data(tmp_path, monkeypatch)
    (root / 'beta' / 'conflict.png').write_bytes((root / 'alpha' / '0.png').read_bytes())
    with pytest.raises(ValueError, match='conflicting labels'):
        data.prepare(root, tmp_path / 'run')


def test_metrics_keep_batch_labels_with_predictions():
    class Tensor:
        def __init__(self, values):
            self.values = np.array(values)
        def numpy(self):
            return self.values
    class Model:
        def __call__(self, images, training=False):
            return Tensor(np.eye(17)[images])
    # Deliberately non-alphabetical batch order: fixed-directory label comparison would fail.
    dataset = [(np.array([5, 1]), Tensor([5, 1])), (np.array([0, 9]), Tensor([0, 9]))]
    result = metrics(Model(), dataset)
    assert result['n'] == 4
    assert result['accuracy'] == 1.0
    assert result['weighted_f1'] == 1.0
    assert len(result['confusion_matrix']) == 17
