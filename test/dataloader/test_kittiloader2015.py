import os

from dataloader.KITTIloader2015 import dataloader


def test_dataloader():
    dataset_folder = '/datasets/kitti15/training/'
    left, right, disp_l, lv, rv, dv = dataloader(dataset_folder)
    assert len(left) == len(right) == len(disp_l)
    assert len(lv) == len(rv) == len(dv)
    for (l, r, d) in zip(left, right, disp_l):
        assert 'image_2' in l
        assert 'image_3' in r
        assert 'disp_occ_0' in d
        assert os.path.basename(l) == os.path.basename(r) == os.path.basename(d)

    for (l, r, d) in zip(lv, rv, dv):
        assert 'image_2' in l
        assert 'image_3' in r
        assert 'disp_occ_0' in d
        assert os.path.basename(l) == os.path.basename(r) == os.path.basename(d)
