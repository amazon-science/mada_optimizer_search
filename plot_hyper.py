import numpy as np
from matplotlib import pyplot as plt

res = []
betas = []
# with open('/fsx/results/hyper_train_log' + '.txt', 'r') as f:
#         res.append(np.genfromtxt(f ,delimiter =', '))

# for i in range(8):
#     with open('/fsx/results/hyper_train_log_rhobeta3_meta_new'+str(i) + '.txt', 'r') as f:
#         res.append(np.genfromtxt(f ,delimiter =', '))

# # with open('/fsx/results/beta_log4' + '.txt', 'r') as f:
# #         betas.append(np.genfromtxt(f ,delimiter =', '))

# res_np = np.vstack(res)
# print(min(res_np[:,4]))
# print(res_np[0])
# print(res_np.shape)
# res_np[np.isnan(res_np)] = 10
# res_np[np.isinf(res_np)] = 10
# res_np.clip(0,2)
# res_np[res_np>1.2] = 1.2
# print(np.max(res_np))

# fig = plt.figure(figsize=[9.6,4.8])
# fig.supylabel('Initial Bet2')
# fig.supxlabel('Initial Beta1')

# plt.subplot(1,2,1)
# plt.tricontourf(res_np[:,0], res_np[:,1], res_np[:,4], levels = 30, vmin = 0.6, vmax = 0.8)
# # plt.xlabel("Initial Beta_1")
# # plt.ylabel("Initial Beta_2")
# plt.yticks(list(np.arange(0.5,1,0.1))+[0.99])
# plt.xticks([0.01]+list(np.arange(0.25,1,0.25))+[0.99])
# plt.title("All 6 parameters are learned")
# # indices = np.arange(0,len(betas[0]),10)
# # mark  =  betas[0][indices,0:2]
# # plt.plot(betas[0][:,0],betas[0][:,1],markevery=90, marker = 'o', color = 'r' )
# print(res_np[:,0])
# print(res_np[:,1])
# print(res_np[:,2])
#plt.show()
#plt.colorbar()
#plt.savefig("contour_hyper_rhobeta3_new.png")

# plt.subplot(1,2,2)
for i in range(8):
    with open('/fsx/results/train_log_mada_3_b3r'+str(i) + '.txt', 'r') as f:
        res.append(np.genfromtxt(f ,delimiter =', '))

# with open('/fsx/results/beta_log4' + '.txt', 'r') as f:
#         betas.append(np.genfromtxt(f ,delimiter =', '))

res_np = np.vstack(res)
print(min(res_np[:,4]))
print(res_np[0])
print(res_np.shape)
# res_np[np.isnan(res_np)] = 10
# res_np[np.isinf(res_np)] = 10
# res_np.clip(0,2)
# res_np[res_np>1.2] = 1.2
# print(np.max(res_np))

# plt.subplot(1,2,2)
plt.tricontourf(res_np[:,2], res_np[:,3], res_np[:,-2], levels = 20), #vmin = 0.6, vmax = 0.8)
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\beta_3$")
plt.yticks(list(np.arange(0.0,1,0.2))+[1.0])
plt.xticks([0.01]+list(np.arange(0.2,1,0.2))+[1.0])
#plt.title("Only Beta1 and Beta2 are learned")
# indices = np.arange(0,len(betas[0]),10)
# mark  =  betas[0][indices,0:2]
# plt.plot(betas[0][:,0],betas[0][:,1],markevery=90, marker = 'o', color = 'r' )
plt.show()
plt.colorbar()
plt.savefig("contour_b3rho_grid.png", dpi=600)