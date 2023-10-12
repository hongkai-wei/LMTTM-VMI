'''
只需要改path为你tensorbarod生成的event即可
调用：  export(tag_names, "要生产的xlsx名字，后缀名不要写错了")

'''
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import tqdm
from config import Config
import os

config = Config.getInstance()["train"]
name = config["name"]

import os

path1 = rf".\log\{name}_test"
directory1 = path1
file_paths1 = []
for root, dirs, files1 in os.walk(directory1):
    for file1 in files1:
        file_path1 = os.path.join(root, file1)
        file_paths1.append(file_path1)

path2 = rf".\log\{name}_train"
directory2 = path2
file_paths2 = []
for root, dirs, files2 in os.walk(directory2):
    for file2 in files2:
        file_path2 = os.path.join(root, file2)
        file_paths2.append(file_path2)

tag_names1 = []
event1 = event_accumulator.EventAccumulator(file_paths1[0])
event1.Reload()
tag_names1 = event1.Tags()['scalars']  # return all tags your named

tag_names2 = []
event2 = event_accumulator.EventAccumulator(file_paths2[0])
event2.Reload()
tag_names2 = event2.Tags()['scalars']  # return all tags your named

def get_val(event: event_accumulator.EventAccumulator, single_tag):
    single_tag_val = event.scalars.Items(single_tag)
    return single_tag_val  # get  single tag 's value data

def export1(tag_list, excel_path):
    writer = pd.ExcelWriter(path=excel_path)
    for sing_tag in tqdm.tqdm(tag_names1):
        temp = get_val(event1, single_tag=sing_tag)
        temp_data = pd.DataFrame(temp)
        data = temp_data.drop("wall_time", axis=1)
        data.to_excel(writer, sheet_name=sing_tag)
    writer.close()

def export2(tag_list, excel_path):
    writer = pd.ExcelWriter(path=excel_path)
    for sing_tag in tqdm.tqdm(tag_names2):
        temp = get_val(event2, single_tag=sing_tag)
        temp_data = pd.DataFrame(temp)
        data = temp_data.drop("wall_time", axis=1)
        data.to_excel(writer, sheet_name=sing_tag)
    writer.close()


if __name__ == "__main__":
    xlsx1 = rf".\log_excel\{name}_test\{name}_test.xlsx"
    xlsx2 = rf".\log_excel\{name}_train\{name}_train.xlsx"
    
    if os.path.exists("./log_excel"):
        pass
    else:
        os.mkdir("./log_excel")

    if os.path.exists(f"./log_excel/{name}_test"):
        pass
    else:
        os.mkdir(f"./log_excel/{name}_test")

    if os.path.exists(f"./log_excel/{name}_train"):
        pass
    else:
        os.mkdir(f"./log_excel/{name}_train")

    export1(tag_names1, xlsx1)
    export2(tag_names2, xlsx2)