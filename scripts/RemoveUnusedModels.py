from os import listdir, remove
from os.path import join

DIR_ROOT = "/home/jupyter/workdir/TaiDoan/Projects/InfoMUNIT_workshop/Models"

if __name__ == '__main__':
    list_projects = [x for x in listdir(DIR_ROOT) if (x != ".ipynb_checkpoints" and "inception" not in x)]

    ## except 2 being trained models
    list_projects.remove("010_MUNIT_origin_dog2cat_64_selective")
    list_projects.remove("011_MUNIT_origin_cityscapes_64")
    list_projects.sort()
    # print(list_projects)

    for project in list_projects:
        path_models = join(DIR_ROOT, project, "outputs")
        outputs = [x for x in listdir(path_models) if x != ".ipynb_checkpoints"]
        assert len(outputs) == 1
        dir_output = outputs[0]
        path_models = join(path_models, dir_output, "checkpoints")
        print(path_models)

        list_checkpoints = [x for x in listdir(path_models) if ("dis_" in x or "gen_" in x)]
        list_checkpoints.sort()
        for checkpoint in list_checkpoints:
            # print(checkpoint)
            iter = checkpoint.replace("dis_", "").replace("gen_", "").replace(".pt", "")
            if int(iter) % 100000 == 0:
                print(checkpoint, "keep")
            else:
                # remove()
                print(checkpoint, "remove")
                remove(join(path_models, checkpoint))
