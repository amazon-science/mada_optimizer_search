import numpy as np
from matplotlib import pyplot as plt


betas = []
# with open('/fsx/results/out_adam/20230907-001652_traj_owt_gpt2_124_adam.txt', 'r') as f:
with open('out_adam/20230917-191731/traj_owt_gpt2_124_adam.txt', 'r') as f:
    res_adam = np.genfromtxt(f ,delimiter =', ')

# with open('/fsx/results/out_mada/20230910-003939_traj_owt_gpt2_124_mada.txt', 'r') as f:
with open('out_mada/20230916-233953/traj_owt_gpt2_124_mada.txt', 'r') as f:
    res_mada = np.genfromtxt(f ,delimiter =', ')

# with open('out_adam/20230911-143630/traj_owt_gpt2_124_adam.txt', 'r') as f:
#     res_mada_final = np.genfromtxt(f ,delimiter =', ')

# with open('out_mada/20230911-142344/traj_owt_gpt2_124_mada.txt', 'r') as f:
#     res_hyperadam = np.genfromtxt(f ,delimiter =', ')

fig = plt.figure()
x = np.arange(0,101)
plt.plot(x,res_adam[:,-1][::100], color = 'b')
plt.plot(x,res_mada[:,-1][::100], color = 'r')
fig.set_size_inches(7,4)

res_adam_np = np.array(res_adam[:,-1])
res_mada_np = np.array(res_mada[:,-1])
# res_mada_final_np = np.array(res_mada_final[:,-1])
# res_hyperadam_np = np.array(res_hyperadam[:,-1])

x_labels = ['0', '20K', '40K', '60K', '80K', '100K']
ticks = [0, 20, 40, 60, 80, 100]

plt.legend(['Adam','Mada','Mada final state','HyperAdam'])
plt.xlabel("Iterations")
plt.ylabel("Validation loss")
plt.ylim([2.85,3.35])
plt.xticks(ticks,labels=x_labels)
plt.grid()
plt.show()
plt.savefig("results/gpt2comp_det.png", dpi=600)