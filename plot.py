import numpy as np
from matplotlib import pyplot as plt

res = []
betas = []
# with open('/fsx/results/train_log' + '.txt', 'r') as f:
#         res.append(np.genfromtxt(f ,delimiter =', '))

for i in range(8):
    with open('/fsx/results/train_log'+str(i) + '.txt', 'r') as f:
        print(i)
        res.append(np.genfromtxt(f ,delimiter =', '))

with open('/fsx/results/beta_traj3' + '.txt', 'r') as f:
        betas.append(np.genfromtxt(f ,delimiter =', '))

res_np = np.vstack(res)
print(min(res_np[:,2]))
print(res_np[0])
print(res_np.shape)
res_np[np.isnan(res_np)] = 10
res_np[np.isinf(res_np)] = 10
res_np.clip(0,2)
res_np[res_np>1.2] = 1.2
print(np.max(res_np))

plt.figure()
#plt.tricontourf(res_np[:,0][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.75,res_np[:,0]>= 0.75)], levels = 30, vmin = 0.6, vmax = 0.8)
plt.tricontourf(res_np[:,0][np.logical_and(res_np[:,1]>= 0.5,True)], res_np[:,1][np.logical_and(res_np[:,1]>= 0.5,True)], res_np[:,2][np.logical_and(res_np[:,1]>= 0.5,True)], levels = 30, vmin = 0.6, vmax = 0.8)
plt.xlabel("Beta_1")
plt.ylabel("Beta_2")
plt.yticks(list(np.arange(0.5,1,0.1))+[0.99])
plt.xticks([0.01]+list(np.arange(0.25,1,0.25))+[0.99])
plt.title("Adam")
indices = np.arange(0,len(betas[0]),10)
mark  =  betas[0][indices,0:2]
#plt.plot(betas[0][:,0],betas[0][:,1],markevery=90, marker = 'o', color = 'r' )
print(res_np[:,0])
print(res_np[:,1])
print(res_np[:,2])
plt.show()
plt.colorbar()
plt.savefig("contour_adam.png")