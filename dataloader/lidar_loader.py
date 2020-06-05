import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    all_files = [os.path.join(filepath, a_file) for a_file in os.listdir(filepath)]
    all_left_img = [img for img in all_files if img.find('left_image') > -1 and is_image_file(img)]
    all_right_img = [img for img in all_files if img.find('right_image') > -1 and is_image_file(img)]
    all_left_disp = [dsp for dsp in all_files if dsp.find('x_disparity') > -1]
    all_left_img.sort()
    all_right_img.sort()
    all_left_disp.sort()
    return all_left_img, all_right_img, all_left_disp
