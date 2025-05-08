import os

lr = 0.1
batch_size = 2**7
num_epochs = 100
momentum = 0.9
weight_decay = 5e-4  
gamma = 0.1

# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.1} --type_t {"real"} --real {"True"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.2} --type_t {"real"} --real {"True"}")  
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.4} --type_t {"real"} --real {"True"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.6} --type_t {"real"} --real {"True"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.8} --type_t {"real"} --real {"True"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {1.0} --type_t {"real"} --real {"True"}")

# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.1} --type_t {"binary"} --real {"False"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.2} --type_t {"binary"} --real {"False"}")  
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.4} --type_t {"binary"} --real {"False"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.6} --type_t {"binary"} --real {"False"}")
# os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {0.8} --type_t {"binary"} --real {"False"}")
os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones 30 50 70 80 --gamma {gamma} --SPC_portion {1.0} --type_t {"binary"} --real {"False"}")