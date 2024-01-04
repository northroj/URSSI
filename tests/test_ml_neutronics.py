import numpy as np
import pytest
from ml_neutronics.testmain import normalize_data

def test_normalize_data():
    input_array1 = np.ones((2,101))
    input_array2 = np.ones((4,101))
    normalized_array = normalize_data(input_array1, input_array2)