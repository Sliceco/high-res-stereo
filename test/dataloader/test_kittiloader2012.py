import os

from dataloader.KITTIloader2012 import dataloader


def test_dataloader():
    dataset_folder = '/datasets/kitti12/training/'
    left, right, disp_l = dataloader(dataset_folder)
    assert len(left) == len(right) == len(disp_l)
    for (l, r, d) in zip(left, right, disp_l):
        assert 'colored_0' in l
        assert 'colored_1' in r
        assert 'disp_occ' in d
        assert os.path.basename(l) == os.path.basename(r) == os.path.basename(d)
    return
