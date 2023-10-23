import os
import json

'''
Create a main.py file under exp as a basic model for finding different high-parameter experiments
'''

os.system("python exp\\train_continual.py best_process_unit_and_memory_mode.json ")
os.system("python exp\\predict_continual.py best_process_unit_and_memory_mode.json")
os.system("python exp\\tesorboard2excel.py best_process_unit_and_memory_mode.json")