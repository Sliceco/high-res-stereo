import os

import cv2
import numpy as np
import pytest

import dataloader.MiddleburyLoader as DA
from dataloader.listsceneflow import dataloader
from utils.preprocess import get_inv_transform


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


def test_daloder():
    show_images = True
    all_left_img, all_right_img, all_left_disp, all_right_disp = dataloader('C:/datasets/sceneflow/')
    lidar_loader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                    rand_scale=[0.9, 2.4], order=2)
    inv_t = get_inv_transform()
    j = 0
    for left_img, right_img, disp in lidar_loader:
        j += 1
        if show_images:
            left_img_np = np.array(inv_t(left_img)).astype(np.uint8)
            right_img_np = np.array(inv_t(right_img)).astype(np.uint8)
            disp_img = cv2.normalize(disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            rectified_pair = np.concatenate((left_img_np, right_img_np), axis=1)
            h, w, _ = rectified_pair.shape
            for i in range(10, h, 30):
                rectified_pair = cv2.line(rectified_pair, (0, i), (w, i), (0, 0, 255))
            cv2.imshow('rectified', rectified_pair[:, :, ::-1])
            cv2.imshow('disp_img', disp_img)
            cv2.waitKey(0)
        assert left_img.shape == (3, 512, 768)
        assert right_img.shape == (3, 512, 768)
        assert disp.shape == (512, 768)
