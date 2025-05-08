import os

lr = 0.1
batch_size = 2**7
num_epochs = 50
momentum = 0.9
weight_decay = 5e-4  
gamma = 0.1
dropouts = [0.1, 0.2, 0.3, 0.5]
#dropouts = [0.1]
milestones = ["10 20 30 40", "15 30 45", "20 40"]
#milestones = ["10 20 30 40"]

os.chdir("baseline")
# Confirmar que el cambio fue exitoso
print("Directorio actual:", os.getcwd())

for milestone in milestones:
    for dropout in dropouts:
        os.system(f"python main_cifar10_base.py --batch_size {batch_size} --num_epochs {num_epochs} --momentum {momentum} --weight_decay {weight_decay} --milestones {milestone} --gamma {gamma} --SPC_portion {0.1} --type_t binary --real False --dropout {dropout}")