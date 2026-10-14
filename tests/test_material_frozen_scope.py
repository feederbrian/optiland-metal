"""Index lookups inside a frozen-materials scope equal lookups outside one.

Inside :func:`frozen_materials_scope` a material fingerprints its state once
and remembers the last wavelength array of each property, so traces skip the
per-lookup fingerprint and cache-key work. Every value must still be exactly
the value computed outside a scope, including after a wavelength array is
written in place, and a material changed between scopes must be re-read.
"""

from __future__ import annotations

import numpy as np
import pytest

from optiland.materials import AbbeMaterial, IdealMaterial, Material
from optiland.materials.base import frozen_materials_scope

QUERIES = {
    "scalar": 0.55,
    "uniform_small": np.full(91, 0.5876),
    "mixed_small": np.linspace(0.45, 0.65, 91),
    "uniform_large": np.full(3000, 0.4861),
    "mixed_large": np.linspace(0.45, 0.65, 3000),
    "float32": np.full(7, 0.55, dtype=np.float32),
}

MATERIALS = {
    "glass": lambda: Material("N-BK7"),
    "ideal": lambda: IdealMaterial(n=1.6, k=0.0),
    "abbe": lambda: AbbeMaterial(n=1.62, abbe=36.4, model="buchdahl"),
}


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("material", sorted(MATERIALS))
def test_values_match_outside_scope(material):
    reference = MATERIALS[material]()
    scoped = MATERIALS[material]()
    expected = {name: reference.n(q) for name, q in QUERIES.items()}
    with frozen_materials_scope():
        for _ in range(3):  # repeated lookups take the remembered path
            for name, query in QUERIES.items():
                assert _same(scoped.n(query), expected[name]), name
                assert _same(scoped.k(query), reference.k(query)), name


def test_in_place_wavelength_write_is_seen():
    glass = Material("N-BK7")
    w = np.full(16, 0.55)
    with frozen_materials_scope():
        first = np.array(glass.n(w))
        w[:] = 0.65
        second = np.array(glass.n(w))
        w[3] = 0.45
        third = np.array(glass.n(w))
    assert _same(first, Material("N-BK7").n(np.full(16, 0.55)))
    assert _same(second, Material("N-BK7").n(np.full(16, 0.65)))
    expected = np.full(16, 0.65)
    expected[3] = 0.45
    assert _same(third, Material("N-BK7").n(expected))


def test_material_change_between_scopes_is_seen():
    ideal = IdealMaterial(n=1.5)
    w = np.full(4, 0.55)
    with frozen_materials_scope():
        before = np.array(ideal.n(w))
    ideal.index[0] = 1.7  # an in-place write between operations
    with frozen_materials_scope():
        after = np.array(ideal.n(w))
    assert np.all(before == 1.5)
    assert np.all(after == 1.7)


def test_nested_scopes_share_the_outer_one():
    glass = Material("N-BK7")
    w = np.full(8, 0.5876)
    with frozen_materials_scope():
        outer = glass.n(w)
        with frozen_materials_scope():
            inner = glass.n(w)
    assert _same(outer, inner)
