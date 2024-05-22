import numpy as np
from matplotlib import pyplot as plt


betas = []
with open('/fsx/results/adam_best.txt', 'r') as f:
    res_adam = np.genfromtxt(f ,delimiter =', ')

with open('/fsx/results/mada_best.txt', 'r') as f:
    res_mada = np.genfromtxt(f ,delimiter =', ')

plt.figure()
#plt.tricontourf(res_np[:,0][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], levels = 30, vmin = 0.6, vmax = 0.8)
#plt.tricontourf(res[:,0][np.logical_and(res[:,1]>= 0.5,True)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.5,True)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.5,True)], levels = 30, vmin = 0.6, vmax = 0.8)
plt.plot(res_adam[:,-1][::25])
plt.plot(res_mada[:,-1][::25])

res_adam_np = np.array(res_adam[:,-1])
res_mada_np = np.array(res_mada[:,-1])

plt.legend(['Adam','Mada'])
plt.xlabel("Iterations")
plt.ylabel("Parameter value")
plt.ylim([3.7,4.5])
#plt.yticks(list(np.arange(0,1.1,0.1)))
#plt.title("h_lr=1e-3 h_mu = 0; h_lr= 1e-2, mu =0.9 for rho,c,gamma, val_loss = 3.7861",fontsize = 8 )
plt.grid()
plt.show()
plt.savefig("adamvsmada2.png", dpi=300)