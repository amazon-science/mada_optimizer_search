import numpy as np
from matplotlib import pyplot as plt

res_adam = []
res_mada = []
res_hadam = []
betas = []

for i in range(8):
    with open('out-shakespeare/train_log_adam'+str(i) + '.txt', 'r') as f:
        res_adam.append(np.genfromtxt(f ,delimiter =', '))
    with open('out-shakespeare/train_log_hyperadam'+str(i) + '.txt', 'r') as f:
        res_hadam.append(np.genfromtxt(f ,delimiter =', '))
    with open('out-shakespeare/hyper_train_log_rhobeta3_finetune_meta_new_25lr'+str(i) + '.txt', 'r') as f:
        res_mada.append(np.genfromtxt(f ,delimiter =', '))


res_np_adam = np.vstack(res_adam)
res_np_adam[np.isnan(res_np_adam)] = 10
res_np_adam[np.isinf(res_np_adam)] = 10

res_np_adam[res_np_adam[:,-2]>0.8,-2] = 0.8

res_np_mada = np.vstack(res_mada)
res_np_mada[np.isnan(res_np_mada)] = 10
res_np_mada[np.isinf(res_np_mada)] = 10

res_np_mada[res_np_mada[:,-2]>0.8,-2] = 0.8

res_np_hadam = np.vstack(res_hadam)
res_np_hadam[np.isnan(res_np_hadam)] = 10
res_np_hadam[np.isinf(res_np_hadam)] = 10

res_np_hadam[res_np_hadam[:,-2]>0.8,-2] = 0.8

fig = plt.figure(figsize=[16,4])

ax1 = plt.subplot(1,3,1)
plt.tricontourf(res_np_adam[:,0], res_np_adam[:,1], res_np_adam[:,-2], levels = 10, vmin=0.6, vmax = 0.8, extend="both")

plt.ylabel(r"Initial $\beta_2$")
plt.yticks(list(np.arange(0.5,1,0.1))+[0.99])
plt.xticks([0.01]+list(np.arange(0.25,1,0.25))+[0.99])


ax2 = plt.subplot(1,3,2)
plt.tricontourf(res_np_mada[:,0], res_np_mada[:,1], res_np_mada[:,-2], levels = 10, vmin = 0.6, vmax = 0.8)
plt.xlabel(r"Initial $\beta_1$")
plt.yticks(list(np.arange(0.5,1,0.1))+[0.99])
plt.xticks([0.01]+list(np.arange(0.25,1,0.25))+[0.99])

ax3 = plt.subplot(1,3,3)
im = plt.tricontourf(res_np_hadam[:,0], res_np_hadam[:,1], res_np_hadam[:,-2], levels = 10, vmin = 0.6, vmax = 0.8)
plt.yticks(list(np.arange(0.5,1,0.1))+[0.99])
plt.xticks([0.01]+list(np.arange(0.25,1,0.25))+[0.99])

plt.show()

fig.colorbar(im, ax=[ax1,ax2,ax3])
plt.savefig("results/contour_comp3_2.png", dpi=900)