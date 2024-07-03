#python 
import torch
import numpy as np
import matplotlib.pylab as plt
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm

import torch.nn as nn
import gdtuo
from gdtuo import Meta
#matplotlib inline 


class oneDlayer(nn.Module):
    def __init__(self, x):
        super(oneDlayer, self).__init__()
        self.x = torch.nn.Parameter(torch.Tensor([x])) # Variable(torch.Tensor([x]), requires_grad=True) # torch.nn.Parameter(custom_weight)

    def forward(self, t):
        if self.x > 1.0:
            self.x = torch.nn.Parameter(torch.Tensor([1.0]))
        if self.x < -1.0:
            self.x = torch.nn.Parameter(torch.Tensor([-1.0]))
        if t%101 == 1:
            return self.x*1010.0
        else:
            return -10.0*self.x

# class oneD(nn.Module):

#     def init(self):
#         super(oneD, self).__init__()
#         self.l1 = oneDlayer(0.5)

#     def forward(self,t):
#         return self.l1(t)


#define torch variables
xmax = Variable(torch.Tensor([1.0]), requires_grad=True)
xmin = Variable(torch.Tensor([-1.0]), requires_grad=True)

# define objective finction
def constrain(x):
    if x > 1.0:
        x = xmax
    if x < -1.0:
        x = xmin
    return x

def ft(x, t):
    if t%101 == 1:
        return x*1010.0
    else:
        return -10.0*x
    
def fmin(t):
    if t%101 == 1:
        return -1010.0
    else:
        return 10.0

T = 2000000
def Ft_OnlineLearning_old(learning_rate=1e-3, amsgrad=True):
    '''
        Online learning experiment
        Input: learning_rate, amsgrad = Ture/False
        Ouput: Ft_step -- time step, 
               Ft_Rt   -- average regret,
               Ft_x    -- value of the iterate
    '''
    x    = Variable(torch.Tensor([1]), requires_grad=True)

    
    optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, amsgrad=amsgrad)
    Rt_sum   = 0
    Ft_step = []
    Ft_Rt   = []
    Ft_x    = []

    for step in tqdm(range(1,T+1)):
        x    = constrain(x)
        loss = ft(x, step)

        Rt_sum += (loss.item() - fmin(step))
        avg_Rt = Rt_sum/step

        if step%1000 == 0:
            print(step, loss.item(), x.item(), avg_Rt)
            Ft_step.append(step)
            Ft_Rt.append(avg_Rt)
            Ft_x.append(x.item())

        # manually zero all previous gradients
        optimizer.zero_grad()
        # calculate new gradients
        loss.backward()
        # apply new gradients
        optimizer.step()    
    
    return Ft_step, Ft_Rt, Ft_x

def Ft_OnlineLearning(learning_rate=1e-3, amsgrad=False):
    '''
        Online learning experiment
        Input: learning_rate, amsgrad = Ture/False
        Ouput: Ft_step -- time step, 
               Ft_Rt   -- average regret,
               Ft_x    -- value of the iterate
    '''
    x    = Variable(torch.Tensor([0.5]), requires_grad=True)

    model = oneDlayer(0)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, amsgrad=amsgrad)
    Rt_sum   = 0
    Ft_step = []
    Ft_Rt   = []
    Ft_x    = []

    for step in tqdm(range(1,T+1)):
        #model.x.data = constrain(model.x.data)
        loss = model(step)

        Rt_sum += (loss.item() - fmin(step))
        avg_Rt = Rt_sum/step
        
        if step%10000 == 0:
            print(step, loss.item(), model.x.data, avg_Rt)
            Ft_step.append(step)
            Ft_Rt.append(avg_Rt)
            Ft_x.append(model.x.item())

        # manually zero all previous gradients
        optimizer.zero_grad()
        # calculate new gradients
        loss.backward()
        # apply new gradients
        optimizer.step()    
    
    return Ft_step, Ft_Rt, Ft_x

def Ft_OnlineLearning_Mada(learning_rate=1e-3, amsgrad=False):
    '''
        Online learning experiment
        Input: learning_rate, amsgrad = Ture/False
        Ouput: Ft_step -- time step, 
               Ft_Rt   -- average regret,
               Ft_x    -- value of the iterate
    '''
    x    = Variable(torch.Tensor([0.5]), requires_grad=True)
    
    

    #optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, amsgrad=amsgrad)
    Rt_sum   = 0
    Ft_step = []
    Ft_Rt   = []
    Ft_x    = []
    Ft_rho = []
    model = oneDlayer(-0.0)#.to('cuda:0')
    optim = gdtuo.Meta(alpha=learning_rate, beta1 = 0.9, beta2 = 0.99, beta3 = 0.0, rho = 1.0, c=1.0, gamma=1.0, 
                       optimizer=gdtuo.SGDPerParamMo(params = [['beta1', 1e-1, 0.5], ['beta2', 1e-1, 0.5], ['beta3', 1e-1, 0.9], ['rho', 1e-1, 0.9], ['c', 1e-1, 0.9], ['gamma', 0.0, 0.0], ['alpha' ,0.0 , 0.0]  ]))
    #optim=gdtuo.Meta(alpha=1e-5, beta1 = 0.9, beta2 = 0.99, beta3 = 0.0, rho = 1.0, optimizer = gdtuo.SGDPerParam(params = [ ['beta1' ,1e-3], ['beta2', 1e-3], ['rho', 1e-1]]))# gdtuo.SGDPerParam(params = [['beta1' ,1e-3], ['beta2', 1e-3], ['rho', 1e-2]])
    #optim = gdtuo.Adam(alpha = 1e-1, beta1 = 0.9, beta2 = 0.99)
    mw = gdtuo.ModuleWrapper(model, optimizer=optim)
    mw.initialize()

    for step in tqdm(range(1,T+1)):

        
        mw.begin()
        loss = mw.forward(step)

        with torch.no_grad():
            Rt_sum += (loss.item() - fmin(step))
            avg_Rt = Rt_sum/step

        if step%1000 == 0:
            print(step, loss.item(), model.x.item(), avg_Rt)
            print(f"beta1 {mw.optimizer.parameters['beta1']:.4f}, beta2 {mw.optimizer.parameters['beta2']:.4f}, beta3 {mw.optimizer.parameters['beta3']:.4f}, rho {mw.optimizer.parameters['rho']:.4f}, c {mw.optimizer.parameters['c']:.4f}, gamma {mw.optimizer.parameters['gamma']:.4f}, alpha {mw.optimizer.parameters['alpha']}")
            Ft_step.append(step)
            Ft_Rt.append(avg_Rt)
            Ft_x.append(model.x.item())
            Ft_rho.append(mw.optimizer.parameters['rho'].item())

        # manually zero all previous gradients
        mw.zero_grad()
        # calculate new gradients
        loss.backward(create_graph=True)
        mw.optimizer.parameters['alpha'].grad = torch.zeros_like(mw.optimizer.parameters['alpha'].grad)
        mw.optimizer.parameters['beta1'].grad = torch.zeros_like(mw.optimizer.parameters['beta1'].grad)
        mw.optimizer.parameters['beta2'].grad = torch.zeros_like(mw.optimizer.parameters['beta2'].grad)
        # mw.optimizer.parameters['beta3'].grad = torch.zeros_like(mw.optimizer.parameters['beta3'].grad)
        # mw.optimizer.parameters['rho'].grad = torch.zeros_like(mw.optimizer.parameters['rho'].grad)
        # mw.optimizer.parameters['c'].grad = torch.zeros_like(mw.optimizer.parameters['c'].grad)
        # mw.optimizer.parameters['gamma'].grad = torch.zeros_like(mw.optimizer.parameters['gamma'].grad)
        # beta3 helps rho go down, c does not spoil it, 
        # apply new gradients
        mw.step()    
    
    return Ft_step, Ft_Rt, Ft_x, Ft_rho

def plot_Mada(MAda_step, MAda_Rt, MAda_x, MAda_rho):
    
    title = 'Learning rate = ' + str(lr)
    plt.figure(figsize=(18,6))
    plt.subplot(1, 2, 1)

    
    
    plt.plot(MAda_step, MAda_Rt,label='Mada')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, 0, 1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Rt/t', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)


    plt.subplot(1, 3, 2)
    plt.plot(MAda_step, MAda_x,label='Mada')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, -1.1, 1.1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('xt', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.subplot(1, 3, 3)
    plt.plot(MAda_step, MAda_rho,label='Mada')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, -1.1, 1.1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('xt', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)


    plt.savefig('results/mada.png')

def plot_Res(Adam_step, Adam_Rt, Adam_x, amsg_step, amsg_Rt, amsg_x, lr):
    
    title = 'Learning rate = ' + str(lr)
    plt.figure(figsize=(24,6))
    plt.subplot(1, 3, 1)

    
    plt.plot(Adam_step, Adam_Rt, label='Adam')
    plt.plot(amsg_step, amsg_Rt,label='AMSGrad')
    plt.plot(MAda_step, MAda_Rt,label='Mada')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, 0, 1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Rt/t', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)


    plt.subplot(1, 3, 2)
    plt.plot(Adam_step, Adam_x, label='Adam')
    plt.plot(amsg_step, amsg_x,label='AMSGrad')
    plt.plot(MAda_step, MAda_x,label='Mada')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, -1.1, 1.1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('xt', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.subplot(1, 3, 3)
    plt.plot(MAda_step, MAda_rho,label='rho')
    plt.grid()
    plt.legend(fontsize=14)
    plt.title(title)
    plt.axis([0,T, -0.1, 1.1])
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('rho_t', fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.savefig('amsgradrho.png', dpi=900)

lr = 1e-3


# # # MADA
MAda_step, MAda_Rt, MAda_x, MAda_rho = Ft_OnlineLearning_Mada(learning_rate=lr)

# # Adam
Adam_step, Adam_Rt, Adam_x = Ft_OnlineLearning(learning_rate=lr, amsgrad=False)

# # # AMSGrad
amsg_step, amsg_Rt, amsg_x = Ft_OnlineLearning(learning_rate=lr, amsgrad=True)

plot_Res(Adam_step, Adam_Rt, Adam_x, amsg_step, amsg_Rt, amsg_x,lr)



