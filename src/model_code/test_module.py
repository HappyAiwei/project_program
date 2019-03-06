import pytest
import numpy as np
import pandas as pd
from module import RMSE

@pytest.fixture
def compute_example():
    y_val = pd.DataFrame(np.array([1, 2, 3, 4]), columns=['a'])
    pred = np.array([3, 4, 5, 6])
    result = RMSE(y_val, pred)
    return result

def test_eq(compute_example):
    fun_result = compute_example
    assert fun_result == 2





