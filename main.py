# 先执行train.py再执行predict.py

import os
os.system("python train.py")
os.system("python predict.py")
os.system("python tesorboard2excel.py")