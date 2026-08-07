import random

import matplotlib
import pytest


matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def deterministic_randomness():
    random.seed(1729)
