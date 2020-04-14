"""
    Crop images to the head of the pet according to the given key points
"""
from os import walk, makedirs
from os.path import join, exists
import cv2
import math
from scipy import ndimage
import numpy as np
from multiprocessing import Pool

DATASET = "cat-dataset"
path_dir_dataset = '/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/dog2cat/CAT_DATASET/CAT_DATASET'
path_dir_out = '/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/dog2cat/CAT_DATASET/Rotated'


def get_crop(_keys):
    _y_min = 9999999
    _y_max = 0
    _x_min = 9999999
    _x_max = 0
    for _key in keys:
        if _key[0] < _y_min:
            _y_min = _key[0]
        if _key[0] > _y_max:
            _y_max = _key[0]
        if _key[1] < _x_min:
            _x_min = _key[1]
        if _key[1] > _x_max:
            _x_max = _key[1]
    return _x_min, _y_min, _x_max, _y_max


def get_rotate_angle(_keys):
    """
        https://stackoverflow.com/questions/2676719/calculating-the-angle-between-the-line-defined-by-two-points
        delta_x = touch_x - center_x
        delta_y = touch_y - center_y
        theta_radians = atan2(delta_y, delta_x)
    """
    delta_x = _keys[1][0] - _keys[0][0]
    delta_y = _keys[1][1] - _keys[0][1]
    print(delta_x)
    print(delta_y)
    theta_radians = math.atan2(delta_y, delta_x)
    return float(theta_radians) * 180.0 / math.pi


def rotate_point(x, y, x_center, y_center, _angle):
    _angle = _angle * math.pi / 180.0
    x_rot = (x - x_center) * math.cos(_angle) - (y - y_center) * math.sin(_angle) + x_center + 6
    y_rot = (x - x_center) * math.sin(_angle) + (y - y_center) * math.cos(_angle) + y_center + 16
    return int(x_rot), int(y_rot)


def color_keys(_keys, _image, color=None):
    new_image = np.copy(_image)
    if color is None:
        color = [0, 0, 255]
    for _key in _keys:
        new_image[_key[1], _key[0], :] = color
        new_image[_key[1]+1, _key[0], :] = color
        new_image[_key[1], _key[0]+1, :] = color
        new_image[_key[1] - 1, _key[0], :] = color
        new_image[_key[1], _key[0] - 1, :] = color
    return new_image


def rotate_keys(_keys, _angle, _image):
    height, width = _image.shape[:2]
    print("x_center", height)
    print("y_center", width)
    _keys_rot = []
    for _k in _keys:
        _keys_rot.append(rotate_point(_k[0], _k[1], int(height/2), int(width/2), -_angle))
    return _keys_rot

def handle_image(path_file):
    """Get the image"""
    image = cv2.imread(path_file)
    print(path_file)
    # print(image.shape)

    """Get key points"""
    if not exists(path_file + ".cat"):
        return
    with open(path_file + ".cat", 'r') as f:
        keys_raw = f.read()
    print(keys_raw)
    keys_raw_split = keys_raw.split(" ")
    keys = []
    """(Left Eye), (Right Eye), (Mouth), (Left Ear-1), (Left Ear-2), (Left Ear-3), (Right Ear-1), (Right Ear-2), (Right Ear-3)"""
    for i in range(1, 19, 2):
        # print(keys_raw_split[i], keys_raw_split[i+1])
        keys.append([int(keys_raw_split[i]), int(keys_raw_split[i + 1])])

    # image = color_keys(keys, image)
    # cv2.imwrite("no_rot.jpg", color_keys(keys, image))

    """Now we need to rotate the image and its key points"""
    angle = get_rotate_angle(keys)
    image_rotated = ndimage.rotate(image, angle)
    cv2.imwrite(join(path_dir_out, path_file.split("/")[-1]), image_rotated, [cv2.IMWRITE_JPEG_QUALITY, 101])


if __name__ == '__main__':
    makedirs(path_dir_out, exist_ok=True)

    """Make list file paths"""
    list_files = []
    for root, dirs, files in walk(path_dir_dataset):
        if len(files) > 0:
            list_files += [join(root, x) for x in files if x[-4:] == ".jpg"]

    list_files.sort()
    print(len(list_files))
    # print(list_files[:20])

    """Loop the list"""

    pool = Pool(6)
    pool.map(handle_image, list_files[2000:])

    # for path_file in list_files:

        # """Get the image"""
        # image = cv2.imread(path_file)
        # print(path_file)
        # # print(image.shape)
        #
        # """Get key points"""
        # if not exists(path_file+".cat"):
        #     continue
        # with open(path_file+".cat", 'r') as f:
        #     keys_raw = f.read()
        # print(keys_raw)
        # keys_raw_split = keys_raw.split(" ")
        # keys = []
        # """(Left Eye), (Right Eye), (Mouth), (Left Ear-1), (Left Ear-2), (Left Ear-3), (Right Ear-1), (Right Ear-2), (Right Ear-3)"""
        # for i in range(1, 19, 2):
        #     # print(keys_raw_split[i], keys_raw_split[i+1])
        #     keys.append([int(keys_raw_split[i]), int(keys_raw_split[i+1])])
        #
        # # image = color_keys(keys, image)
        # # cv2.imwrite("no_rot.jpg", color_keys(keys, image))
        #
        # """Now we need to rotate the image and its key points"""
        # angle = get_rotate_angle(keys)
        # image_rotated = ndimage.rotate(image, angle)
        # cv2.imwrite(join(path_dir_out, path_file.split("/")[-1]), image_rotated, [cv2.IMWRITE_JPEG_QUALITY, 101])

        # keys_rotated = rotate_keys(keys, angle, image)
        # """Rotate the key points (pain in the ass)"""

        # image_rotated = color_keys(keys, image_rotated)
        # image_rotated = color_keys(keys_rotated, image_rotated, color=[0, 255, 0])
        # cv2.imwrite("rot.jpg", image_rotated)
        # print("angle", angle)
        # exit()

        # height_eye_to_noise = max(keys[2][1] - keys[0][1], keys[2][1] - keys[1][1])
        # assert height_eye_to_noise > 0
        #
        # x_min, y_min, x_max, y_max = get_crop(keys)
        # image = image[
        #         max(x_min - int(height_eye_to_noise/4), 0):min(x_max + height_eye_to_noise, image.shape[0]),
        #         max(y_min - int(height_eye_to_noise/4), 0):min(y_max + int(height_eye_to_noise/4), image.shape[1]), :]
        #
        # cv2.imwrite(join(path_dir_out, path_file.split("/")[-1]), image, [cv2.IMWRITE_JPEG_QUALITY, 101])
        # exit()


