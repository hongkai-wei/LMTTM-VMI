# 先执行train.py再执行predict.py
import os
import json

# 读取JSON文件
with open('./config/base.json', 'r') as file:
    data = json.load(file)

# 修改JSON数据
# data['train']['name'] = 'TTM_MHA_transformer_WritePosition_Load_laplace_mem2048_tokens4'
# data['model']['num_tokens'] = 4

# 写入修改后的JSON文件
with open('./config/base.json', 'w') as file:
    json.dump(data, file, indent=4)

os.system("python train.py")
os.system("python predict.py")
os.system("python tesorboard2excel.py")