import os
import os.path
import sys
from cgi import test

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from src.data.make_dataset import MNISTdata
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_raw_data():
    train_paths = [f"data/raw/corruptmnist/train_{i}.npz" for i in range(5)]

    X_train = np.concatenate(
        [np.load(train_file)["images"] for train_file in train_paths]
    )
    Y_train = np.concatenate(
        [np.load(train_file)["labels"] for train_file in train_paths]
    )

    X_test = np.load("data/raw/corruptmnist/test.npz")["images"]
    Y_test = np.load("data/raw/corruptmnist/test.npz")["labels"]

    N_train = 25000
    N_test = 5000

    assert len(X_train) == N_train, "Dataset did not have the correct number of samples"
    assert len(X_test) == N_test

    assert all(
        img.shape == (28, 28) for img in X_test
    ), "Invalid image shape"  # assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
    assert set(Y_test).issuperset(
        set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ), "Not all classes are covered"  # assert that all labels are represented
