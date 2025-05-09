import os

SPC_portion = 0.15

os.system(f"python main_cifar10_Base.py --milestones 20 40 --SPC_portion {SPC_portion}")