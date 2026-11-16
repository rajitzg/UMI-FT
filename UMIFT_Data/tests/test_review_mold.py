import numpy as np

from review_fixtures import UMIFT_DATA  # noqa: F401  (sets sys.path)
from review import mold


def _write_field(path, n_grid=40, apron_rows=5):
    # flat grid in x/z at y=0; the last `apron_rows` rows are apron
    xs, zs = np.meshgrid(np.linspace(0, 1, n_grid), np.linspace(0, 0.2, n_grid))
    v = np.stack([xs.ravel(), np.zeros(xs.size), zs.ravel()], 1)
    faces = []
    for r in range(n_grid - 1):
        for c in range(n_grid - 1):
            i = r * n_grid + c
            faces += [[i, i + 1, i + n_grid], [i + 1, i + n_grid + 1, i + n_grid]]
    apron = np.zeros(len(v), bool)
    apron[-apron_rows * n_grid:] = True
    np.savez(path, vertices=v, faces=np.asarray(faces), is_apron=apron)
    return len(faces)


def test_load_mold_drops_apron(tmp_path):
    f = tmp_path / "fixb_field.npz"
    n_faces = _write_field(f)
    v, faces = mold.load_mold(f, tmp_path / "cache", max_tris=10**6)
    assert v.dtype == np.float32 and faces.dtype == np.uint32
    assert 0 < len(faces) < n_faces
    assert faces.max() < len(v)
    assert v[:, 2].max() < 0.2 * (40 - 5) / 39 + 1e-6  # no apron vertex survives


def test_load_mold_decimates_and_caches(tmp_path):
    f = tmp_path / "fixb_field.npz"
    _write_field(f)
    v, faces = mold.load_mold(f, tmp_path / "cache", max_tris=500)
    assert len(faces) <= 500
    assert list((tmp_path / "cache").glob("mold_*.npz"))
    v2, faces2 = mold.load_mold(f, tmp_path / "cache", max_tris=500)
    assert np.array_equal(faces, faces2)
