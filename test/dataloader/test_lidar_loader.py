import os

import pytest
import numpy as np
import cv2

from dataloader.lidar_loader import dataloader

from dataloader import MiddleburyLoader as DA
from utils.preprocess import get_inv_transform


@pytest.mark.parametrize('base_path', ['C:/datasets/lidar_dataset/train',
                                       'C:/datasets/lidar_dataset/test',
                                       'C:/datasets/lidar_dataset/validation'])
def test_dataloader(base_path):
    left, right, disp_l = dataloader(base_path)
    assert len(left) == len(right) == len(disp_l)
    i = 0
    for (l, r, dl) in zip(left, right, disp_l):
        if i == 0:
            assert os.path.exists(l)
            assert os.path.exists(r)
            assert os.path.exists(dl)
            i += 1
        assert 'left_image' in l
        assert 'right_image' in r
        assert 'x_disparity' in dl
        l_image_name = os.path.basename(l).split('-')[0]
        r_image_name = os.path.basename(r).split('-')[0]
        disp_image_name = os.path.basename(dl).split('-')[0]
        assert l_image_name == r_image_name == disp_image_name

        disp_data = np.load(dl)['x_disparity']
        min_allowed_disp = (0 - 1e-6)
        assert np.min(disp_data) >= min_allowed_disp
        disp_img = cv2.normalize(disp_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.imshow('disp_img', disp_img)
        cv2.waitKey(0)


# @pytest.mark.skip('Dev visualization test only')
def test_with_torch_dataloader():
    show_images = True
    all_left_img, all_right_img, all_left_disp = dataloader('C:/datasets/lidar_dataset/train')
    lidar_loader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 1.1],
                                    rand_bright=[0.8, 1.2], order=2)
    inv_t = get_inv_transform()
    for left_img, right_img, disp in lidar_loader:
        if show_images:
            left_img_np = np.array(inv_t(left_img)).astype(np.uint8)
            right_img_np = np.array(inv_t(right_img)).astype(np.uint8)
            disp_img = cv2.normalize(disp, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            cv2.imshow('left_img', left_img_np[:, :, ::-1])
            cv2.imshow('right_img', right_img_np[:, :, ::-1])
            cv2.imshow('disp_img', disp_img)
            cv2.waitKey(0)
        assert left_img.shape == (3, 512, 768)
        assert right_img.shape == (3, 512, 768)
        assert disp.shape == (512, 768)

        assert np.min(disp) >= (0 - 1e-6)
        break
