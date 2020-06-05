"""
    Splitting pairs of images from pix2pix dataset
"""

from os import listdir, makedirs
from os.path import join
import cv2

# DIR_IN = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/cityscapes/val"
# DIR_OUT_A = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testA"
# DIR_OUT_B = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/testB"

DIR_IN = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/cityscapes/train"
DIR_OUT_A = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/trainA"
DIR_OUT_B = "/media/tai/6TB/Projects/InfoMUNIT/Data/CityScapes/split/trainB"
SIZE_ONE_IMAGE = 256

if __name__ == '__main__':
    makedirs(DIR_OUT_A, exist_ok=True)
    makedirs(DIR_OUT_B, exist_ok=True)
    list_names = listdir(DIR_IN)
    for file_name in list_names:
        image = cv2.imread(join(DIR_IN, file_name))
        imageA = image[:, SIZE_ONE_IMAGE:, :]
        imageB = image[:, :SIZE_ONE_IMAGE, :]
        print(image.shape)
        print(imageA.shape)
        print(imageB.shape)
        cv2.imwrite(join(DIR_OUT_A, file_name), imageA, [cv2.IMWRITE_JPEG_QUALITY, 101])
        cv2.imwrite(join(DIR_OUT_B, file_name), imageB, [cv2.IMWRITE_JPEG_QUALITY, 101])
