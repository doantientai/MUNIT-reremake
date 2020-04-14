"""
    List down breeds
    Keep the largest breeds and remove the rest
"""

from os import makedirs, listdir
from os.path import join
from shutil import copyfile

DIR_IN = "/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/dog2cat/DataReady/trainA"
DIR_OUT = "/media/tai/6TB/Projects/InfoMUNIT/Data/ver_workshop/dog2cat/DataReadySelective/trainA"

if __name__ == '__main__':
    makedirs(DIR_OUT, exist_ok=True)
    list_names = listdir(DIR_IN)
    list_names.sort()
    print("len list_names:", len(list_names))

    ### count samples in each breed
    breeds_dict = {}
    for file_name in list_names:
        breed = file_name.split("_")[0]
        if breed not in breeds_dict.keys():
            breeds_dict[breed] = 1
        else:
            breeds_dict[breed] += 1

        # print("file_name", file_name)
        # print("breed", breed)
    total_from_large_breeds = 0
    list_breeds_kept = []
    for key, value in breeds_dict.items():
        if value >= 130:
            print(key, value)
            list_breeds_kept.append(key)
            total_from_large_breeds += value

    # print(len(breeds_dict.keys()))

    print('total_from_large_breeds', total_from_large_breeds)

    for file_name in list_names:
        breed = file_name.split("_")[0]
        if breed in list_breeds_kept:
            copyfile(join(DIR_IN, file_name),
                     join(DIR_OUT, file_name))
