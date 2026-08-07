import kiwicalc
import pytest


@pytest.mark.parametrize('name', kiwicalc.__all__)
def test_every_declared_public_export_exists(name):
    assert hasattr(kiwicalc, name), f'kiwicalc.__all__ exports missing name {name!r}'


def test_public_exports_do_not_contain_duplicates():
    assert len(kiwicalc.__all__) == len(set(kiwicalc.__all__))
