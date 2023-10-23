import os
import json

'''
Explanation:
This script selects the best model for positional_embedding.
According to the results, exp2 has the best result, and we choose exp2 as the optimal model parameter.

The best parameter:
                    positional_embedding: Read_use_positional_embedding
'''

####exp1
with open('./config/best_positional_embedding.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp1_None_positional_embedding"
data['model']['Read_use_positional_embedding'] = False
data['model']['Write_use_positional_embedding'] = False

with open('./config/best_positional_embedding.json', 'w') as file:
    json.dump(data, file, indent=4)
    
os.system("python exp\\train_continual.py best_positional_embedding.json ")
os.system("python exp\\predict_continual.py best_positional_embedding.json")
os.system("python exp\\tesorboard2excel.py best_positional_embedding.json")

####exp2
with open('./config/best_positional_embedding.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp2_Read_positional_embedding"
data['model']['Read_use_positional_embedding'] = True
data['model']['Write_use_positional_embedding'] = False
with open('./config/best_positional_embedding.json', 'w') as file:
    json.dump(data, file, indent=4)
os.system("python exp\\train_continual.py best_positional_embedding.json ")
os.system("python exp\\predict_continual.py best_positional_embedding.json")
os.system("python exp\\tesorboard2excel.py best_positional_embedding.json")


####exp3
with open('./config/best_positional_embedding.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp1_Write_positional_embedding"
data['model']['Read_use_positional_embedding'] = False
data['model']['Write_use_positional_embedding'] = True

with open('./config/best_positional_embedding.json', 'w') as file:
    json.dump(data, file, indent=4)

os.system("python exp\\train_continual.py best_positional_embedding.json ")
os.system("python exp\\predict_continual.py best_positional_embedding.json")
os.system("python exp\\tesorboard2excel.py best_positional_embedding.json")

####exp4
with open('./config/best_positional_embedding.json', 'r') as file:
    data = json.load(file)

data['train']['name'] = "exp4_ReadWrite_positional_embedding"
data['model']['Read_use_positional_embedding'] = True
data['model']['Write_use_positional_embedding'] = True

with open('./config/best_positional_embedding.json', 'w') as file:
    json.dump(data, file, indent=4)

os.system("python exp\\train_continual.py best_positional_embedding.json ")
os.system("python exp\\predict_continual.py best_positional_embedding.json")
os.system("python exp\\tesorboard2excel.py best_positional_embedding.json")
