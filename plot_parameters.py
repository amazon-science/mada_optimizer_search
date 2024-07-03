import numpy as np
from matplotlib import pyplot as plt


betas = []
# with open('out_mada/20230916-233953/traj_owt_gpt2_124_mada.txt', 'r') as f:
with open('out_mada/20230910-003939/traj_owt_gpt2_124_mada.txt', 'r') as f:
# with open('out_mada/20230913-193726/traj_owt_gpt2_124_mada.txt', 'r') as f:
    res = np.genfromtxt(f ,delimiter =', ')

    

plt.figure()
x_labels = ['0', '20K', '40K', '60K', '80K', '100K']
ticks = [0, 20, 40, 60, 80, 100]
plt.plot(res[:,0][::100])
plt.plot(res[:,1][::100])
plt.plot(res[:,2][::100])
plt.plot(res[:,3][::100])
plt.plot(res[:,4][::100])
plt.plot(res[:,5][::100])
plt.legend([r'$\beta_1$',r'$\beta_2$', r'$\beta_3$', r'$\rho$', r'c', r'$\gamma$'])
plt.xlabel("Iterations", fontsize = 14)
plt.ylabel("Parameter value", fontsize = 14)
plt.yticks(list(np.arange(0,1.1,0.2)), fontsize = 14)
plt.xticks(ticks,labels=x_labels, fontsize = 14)
plt.grid()
plt.show()
plt.savefig("results/parameters_gpt2_3.png", dpi=900)