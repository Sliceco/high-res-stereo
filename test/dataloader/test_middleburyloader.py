import numpy as np
import cv2

from dataloader import MiddleburyLoader as DA
from dataloader import listfiles as ls

from utils.preprocess import get_inv_transform


def test_myimagefloder():
    show_images = True
    scale_factor = 1.0
    dataset_folder = 'C:/datasets/hrvs/carla-highres/trainingF'
    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(dataset_folder)
    loader_eth3 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                   rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)
    inv_t = get_inv_transform()
    for left_img, right_img, disp in loader_eth3:
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
        break
