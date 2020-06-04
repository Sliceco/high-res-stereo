import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, use_monkaa=False, use_flying=False, use_driving=True):
    all_files = []
    for root, _, files in os.walk(filepath):
        all_files += [os.path.join(root, a_file) for a_file in files]

    images = [img for img in all_files if img.find('frames_cleanpass') > -1 and img[-4:] in IMG_EXTENSIONS]
    disparities = [dsp for dsp in all_files if dsp.find('disparity') > -1]

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []

    # TODO - These are not used by use_flying path could use it
    test_left_img = []
    test_right_img = []
    test_left_disp = []
    test_right_disp = []

    if use_monkaa:
        raise NotImplementedError('Still need to do this part over')

    if use_flying:
        raise NotImplementedError('Still need to do this part over')

    if use_driving:
        all_left_img += [img for img in images if img.find('left') > -1]
        all_right_img += [img for img in images if img.find('right') > -1]
        all_left_disp += [img for img in disparities if img.find('left') > -1]
        all_right_disp += [img for img in disparities if img.find('right') > -1]
    return all_left_img, all_right_img, all_left_disp, all_right_disp
