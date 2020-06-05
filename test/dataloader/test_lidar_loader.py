import os

import pytest
import numpy as np
import cv2

from dataloader.lidar_loader import dataloader

from dataloader import MiddleburyLoader as DA
from utils.preprocess import get_inv_transform


@pytest.mark.parametrize('base_path', ['C:/datasets/lidar_dataset/train',
                                       '/datasets/lidar_dataset/test',
                                       '/datasets/lidar_dataset/validation'])
def test_dataloader(base_path):
    left, right, disp_l = dataloader(base_path)
    max_disp = -1
    min_disp = 100
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
        max_disp = max(max_disp, np.max(disp_data))
        min_disp = min(min_disp, np.min(disp_data))
        assert np.min(disp_data) >= min_allowed_disp
        if np.max(disp_data) > 691:
            disp_img = cv2.normalize(disp_data, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            print(l)
            cv2.imshow('left', cv2.imread(l)[:, ::-1])
            cv2.imshow('right', cv2.imread(r)[:, ::-1])
            cv2.imshow('disp_img', disp_img)
            cv2.waitKey(0)
    print('max_disp: {}'.format(max_disp))
    print('min_disp: {}'.format(min_disp))


# @pytest.mark.skip('Dev visualization test only')
def test_with_torch_dataloader():
    show_images = True
    all_left_img, all_right_img, all_left_disp = dataloader('/datasets/lidar_dataset/')
    lidar_loader = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=[0.9, 1.1],
                                    rand_bright=[0.8, 1.2], order=2)
    inv_t = get_inv_transform()
    for left_img, right_img, disp in lidar_loader:
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
        break
