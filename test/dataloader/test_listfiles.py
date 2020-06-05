import os

import pytest

from dataloader.listfiles import dataloader


@pytest.mark.parametrize('dataset_folder', ['/datasets/eth3d',
                                            '/datasets/middlebury/mb-ex-training/trainingF',
                                            '/datasets/hrvs/carla-highres/trainingF'])
def test_dataloader(dataset_folder):
    left, right, disp_l, disp_r = dataloader(dataset_folder)
    assert len(left) > 0
    assert len(left) == len(right) == len(disp_l) == len(disp_r)
    i = 0
    for (l, r, dl, dr) in zip(left, right, disp_l, disp_r):
        if i == 0:
            assert os.path.exists(l)
            assert os.path.exists(r)
            assert os.path.exists(dl)
            i += 1
        assert os.path.dirname(l) == os.path.dirname(r) == os.path.dirname(dl) == os.path.dirname(dr)
        assert 'im0' in os.path.basename(l)
        assert 'im1' in os.path.basename(r)
        assert 'disp0GT' in os.path.basename(dl)
        assert 'disp1GT' in os.path.basename(dr)
