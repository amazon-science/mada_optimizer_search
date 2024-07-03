import os
import sys
import numpy as np


device = int(sys.argv[1])
step1 = 0.05
step2 = 0.05


if device== 0:
    beta1_list_start = 0.00
    beta1_list_end = 0.125
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 1:
    beta1_list_start = 0.125
    beta1_list_end = 0.25
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 2:
    beta1_list_start = 0.25
    beta1_list_end = 0.375
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 3:
    beta1_list_start = 0.375
    beta1_list_end = 0.5
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 4:
    beta1_list_start = 0.5
    beta1_list_end = 0.625
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 5:
    beta1_list_start = 0.625
    beta1_list_end = 0.75
    beta2_list_start = 0.5
    beta2_list_end = 1.025
elif device== 6:
    beta1_list_start = 0.75
    beta1_list_end = 0.875
    beta2_list_start = 0.5 ##########
    beta2_list_end = 1.025
elif device== 7:
    beta1_list_start = 0.875
    beta1_list_end = 1.05
    beta2_list_start = 0.5 ##############
    beta2_list_end = 1.025


beta1_list = np.arange(beta1_list_start,beta1_list_end,step1)
beta2_list = np.arange(beta2_list_start,beta2_list_end,step2)

beta1_list = np.clip(beta1_list,0.01,0.99)
beta2_list = np.clip(beta2_list,0.01,0.99)
print(beta1_list)
print(beta2_list)

for i in beta1_list:
    for j in beta2_list:

        command = "python train.py config/train_shakespeare_char.py "  + "--beta1=" + str(i) + " --beta2=" + str(j) + " --device=cuda:" + str(device) + " --dtype='float32'" + " --hyperadam=False" + " --adam=False"
        print(command)
        os.system(command)

