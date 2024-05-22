import numpy as np
from matplotlib import pyplot as plt


betas = []
with open('/fsx/results/parameters'+str(0) + '.txt', 'r') as f:
    res = np.genfromtxt(f ,delimiter =', ')

plt.figure()
#plt.tricontourf(res_np[:,0][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], levels = 30, vmin = 0.6, vmax = 0.8)
#plt.tricontourf(res[:,0][np.logical_and(res[:,1]>= 0.5,True)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.5,True)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.5,True)], levels = 30, vmin = 0.6, vmax = 0.8)
plt.plot(res[:,0])
plt.plot(res[:,1])
plt.plot(res[:,2])
plt.plot(res[:,3])
plt.plot(res[:,4])
plt.plot(res[:,5])
plt.legend(['Beta1','Beta2', 'Beta3', 'rho', 'c', 'gamma'])
plt.xlabel("Iterations")
plt.ylabel("Parameter value")
plt.yticks(list(np.arange(0,1.1,0.1)))
plt.title("h_lr=1e-3 h_mu = 0; h_lr= 1e-2, mu =0.9 for rho,c,gamma, val_loss = 3.7861",fontsize = 8 )
plt.grid()
plt.show()
plt.savefig("parameters0.png", dpi=300)