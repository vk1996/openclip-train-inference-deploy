import os
os.chdir("/home/clip-training/")
os.system("mlflow ui --host 0.0.0.0 & python3 train.py")


