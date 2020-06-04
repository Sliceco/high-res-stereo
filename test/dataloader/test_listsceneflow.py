import os

import pytest

from dataloader.listsceneflow import dataloader


@pytest.mark.parametrize(('dataset_folder', 'use_driving'), [('C:/datasets/sceneflow/', True)])
def test_dataloader(dataset_folder, use_driving):
    left, right, disp_l, disp_r = dataloader(dataset_folder, use_driving=use_driving)
    assert len(left) == len(right) == len(disp_l) == len(disp_r)
    i = 0
    for (l, r, dl, dr) in zip(left, right, disp_l, disp_r):
        if i == 0:
            assert os.path.exists(l)
            assert os.path.exists(r)
            assert os.path.exists(dl)
            assert os.path.exists(dr)
            i += 1
        assert os.path.basename(l) == os.path.basename(r)
        assert os.path.basename(dl)[0:-3] == os.path.basename(l)[0:-3]
        assert os.path.basename(dr)[0:-3] == os.path.basename(r)[0:-3]
