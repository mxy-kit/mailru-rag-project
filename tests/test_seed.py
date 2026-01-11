import random
from src.utils import set_seed

def test_seed_reproducible():
    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    b = [random.random() for _ in range(3)]
    assert a == b
