'''
只需要改path为你tensorbarod生成的event即可
调用：  export(tag_names, "要生产的xlsx名字，后缀名不要写错了")

'''
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import numpy as np
import tqdm
path = r"F:\git_ttm\logger\module_have_pos_log_version2.0\events.out.tfevents.1695627864.DESKTOP-MF8V30Q.51008.0"
tag_names = []
event = event_accumulator.EventAccumulator(path)
event.Reload()
tag_names = event.Tags()['scalars']  # return all tags your named


def get_val(event: event_accumulator.EventAccumulator, single_tag):
    single_tag_val = event.scalars.Items(single_tag)
    return single_tag_val  # get  single tag 's value data


def export(tag_list, excel_path):
    writer = pd.ExcelWriter(path=excel_path)
    for sing_tag in tqdm.tqdm(tag_names):
        temp = get_val(event, single_tag=sing_tag)
        temp_data = pd.DataFrame(temp)
        data = temp_data.drop("wall_time", axis=1)
        data.to_excel(writer, sheet_name=sing_tag)
    writer.close()


xlsx = r"F:\git_ttm\logger\module_have_pos_log_version2.0\result.xlsx"

if __name__ == "__main__":
    export(tag_names, xlsx)
