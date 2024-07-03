import torch

class Optimizable:
    '''
    This is the interface for anything that has parameters that need to be
    optimized, somewhat like torch.nn.Model but with the right plumbing for
    hyperoptimizability. (Specifically, torch.nn.Model uses the Parameter
    interface which does not give us enough control about the detachments.)
    Nominal operation of an Optimizable at the lowest level is as follows:
        o = MyOptimizable(...)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = --compute loss function from parameters--
            loss.backward()
            o.step()
        }
    Optimizables recursively handle updates to their optimiz*ers*.
    '''
    def __init__(self, parameters, optimizer):
        self.parameters = parameters # a dict mapping names to tensors
        # if 'transformer.wte.weight' in self.parameters:
        #     self.parameters['transformer.wte.weight'] = self.parameters['lm_head.weight']
        self.optimizer = optimizer   # which must itself be Optimizable!
        self.all_params_with_gradients = []

    def initialize(self):
        ''' Initialize parameters, e.g. with a Kaiming initializer. '''
        pass
    
    def begin(self):
        ''' Enable gradient tracking on current parameters. '''
        # for param in self.all_params_with_gradients:
        #      param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_() # keep gradient information...
            param.retain_grad()    # even if not a leaf...
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        ''' Set all gradients to zero. '''
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    ''' Note: at this point you would probably call .backwards() on the loss
    function. '''

    def step(self):
        ''' Update parameters '''
        pass

    def fake_step(self):
        ''' Update parameters '''
        pass

class NoOpOptimizer(Optimizable):
    '''
    NoOpOptimizer sits on top of a stack, and does not affect what lies below.
    '''
    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def step(self, params):
        pass

    def fake_step(self, params):
        pass

    def __str__(self):
        return ''

class SGD(Optimizable):
    '''
    A hyperoptimizable SGD.
    '''
    def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu)
        }
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            # g = param.grad
            # p = param
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = g
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + g
                g = self.state[name] = buf
            params[name] = (p - g * self.parameters['alpha'])
        
    def __str__(self):
        return 'sgd / '+ str(self.optimizer)

class SGDPerParam(Optimizable):
    '''
    Optimizes parameters individually with SGD.
    '''
    def __init__(self, params, optimizer=NoOpOptimizer()):
        parameters = {k + '_alpha' : torch.tensor(v) for k, v in params}
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if name + '_alpha' not in self.parameters: params[name] = p
            else: params[name] = p - g * self.parameters[name + '_alpha']

    def __str__(self):
        return 'sgdPerParam / ' + str(self.optimizer)

class SGDPerParamMo(Optimizable):
    '''
    Optimizes parameters individually with SGD.
    '''
    def __init__(self, params, optimizer=NoOpOptimizer()):
        parameters = {k + '_alpha' : torch.tensor(v) for k, v, mu in params}
        for k, v, mu in params:
            parameters[k+ '_mu'] = torch.tensor(mu)

        self.state = {}
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if name + '_alpha' not in self.parameters: params[name] = p
            else:
                if  self.parameters[name + '_mu'] != 0.0:
                    if name not in self.state:
                        buf = self.state[name] = g
                    else:
                        buf = self.state[name].detach()
                        buf = buf * self.parameters[name + '_mu'] + g
                    g = self.state[name] = buf
                params[name] = p - g * self.parameters[name + '_alpha']
            
    def __str__(self):
        return 'sgdPerParam / ' + str(self.optimizer)

class AdaGrad(Optimizable):
    '''
    A hyperoptimizable AdaGrad.
    '''
    def __init__(self, alpha=0.01, optimizer=NoOpOptimizer()):
        self.eps = 1e-10
        self.cache = {}
        parameters = {
            'alpha': torch.tensor(alpha)
        }
        super().__init__(parameters, optimizer)
    
    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'G': torch.zeros_like(param) + 1e-1
                }
            g = param.grad.detach()
            self.cache[name]['G'] = G = self.cache[name]['G'].detach() + torch.square(g)
            params[name] = param.detach() - self.parameters['alpha'] * g / torch.sqrt(G + self.eps).detach()
    
    def __str__(self):
        return 'adagrad / ' + str(self.optimizer)

class RMSProp(Optimizable):
    '''
    A hyperoptimizable RMSProp.
    '''
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
            'gamma': RMSProp.unclamp(torch.tensor(gamma))
        }
        super().__init__(parameters, optimizer)
        self.cache = {}

    def step(self, params):
        self.optimizer.step(self.parameters)
        gamma = RMSProp.clamp(self.parameters['gamma'])
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            self.cache[name]['s'] = s = gamma * self.cache[name]['s'].detach() + (1. - gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)
    
    def __str__(self):
        return 'rmsprop / ' + str(self.optimizer)

class RMSPropAlpha(Optimizable):
    '''
    A hyperoptimizable RMSProp for only alpha.
    '''
    def __init__(self, alpha=0.01, gamma=0.99, optimizer=NoOpOptimizer()):
        self.eps = 1e-8
        self.gamma = gamma
        parameters = {
            'alpha': torch.sqrt(torch.tensor(alpha)),
        }
        super().__init__(parameters, optimizer)
        self.cache = {}

    def step(self, params):
        self.optimizer.step(self.parameters)
        alpha = torch.square(self.parameters['alpha'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    's': torch.zeros_like(param)
                }
            g = param.grad.detach()
            self.cache[name]['s'] = s = self.gamma * self.cache[name]['s'].detach() + (1. - self.gamma) * torch.square(g)
            self.all_params_with_gradients.append(s)
            params[name] = param.detach() - alpha * g / torch.sqrt(s + self.eps)
    
    def __str__(self):
        return 'rmspropAlpha / ' + str(self.optimizer)

class Meta(Optimizable):
    '''
    A hyperoptimizable Meta optimizer.
    '''
    
    def clamp(x, min_ = 0.01, max_ = 0.99 ):

        if x>=max_:
            x.data = max_*torch.ones_like(x)
        elif x<=min_:
            x.data = min_*torch.ones_like(x)
        return x


    def unclamp(y):
        return y
    

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.99, beta3=0.0, rho = 1.0, c = 1.0, gamma = 1.0, eps=1e-6, optimizer=NoOpOptimizer()):
        self.eps = eps
        parameters = {
            'alpha': torch.tensor(alpha),
            'beta1': Meta.unclamp(torch.tensor(beta1)),
            'beta2': Meta.unclamp(torch.tensor(beta2)),
            'beta3': Meta.unclamp(torch.tensor(beta3)),
            'rho': Meta.unclamp(torch.tensor(rho)),
            'c': Meta.unclamp(torch.tensor(c)),
            'gamma': Meta.unclamp(torch.tensor(gamma))
        }
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        
        #clamp variables
        beta1 = Meta.clamp(self.parameters['beta1'], min_ = 0.01, max_ = 0.99)
        beta2 = Meta.clamp(self.parameters['beta2'], min_ = 0.501, max_ = 0.99) # min 0.501 for GPT
        beta3 = Meta.clamp(self.parameters['beta3'], min_ = 0.00, max_ = 0.99)
        rho = Meta.clamp(self.parameters['rho'], min_ = 0.0, max_ = 1.0)
        c = Meta.clamp(self.parameters['c'], min_ = 0.0, max_ = 1.0)
        gamma = Meta.clamp(self.parameters['gamma'], min_ = 0.0, max_ = 1.0)
        beta1_lion = 0.95 #parameters suggested for lion for GPT-2
        beta2_lion = 0.98

        param_cache = {}

        for name, param in params.items():
            if id(param) in param_cache:
                params[name] = param_cache[id(param)]
                continue

            if param.grad is None:
                print("none", name)
                continue
                
            g = param.grad.detach()
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) + 1e-15,
                    'v_mean': torch.zeros_like(param)+ 1e-15, #+\
                    'n': g * g,
                    'g': torch.zeros_like(param),
                    'm_lion': torch.zeros_like(param),
                    'u_lion': torch.zeros_like(param)
                }
            
            
            g_hat = g + beta3 * (g - self.cache[name]['g'])
            g_tilde_sq = c * g_hat * g_hat + (1-c)*(self.cache[name]['v'].detach() + g_hat * g_hat * torch.sign(g_hat * g_hat - self.cache[name]['v'].detach()))
            
            self.cache[name]['m'].detach()
            self.cache[name]['v'].detach()
            self.cache[name]['n'].detach()
            self.cache[name]['v_mean'].detach()
            self.cache[name]['g'].detach()

            self.cache[name]['m'] = beta1 * self.cache[name]['m'].data + (1. - beta1) * g.clone()
            self.cache[name]['v'] = beta2 * self.cache[name]['v'].data +\
                (1. - beta2) * (g_tilde_sq)
            del g_tilde_sq
            self.cache[name]['n'] = beta3 * self.cache[name]['n'].data + (1. - beta3) * (g - self.cache[name]['g'].data)
            self.cache[name]['g'] = g
            self.cache[name]['m_lion'] = m_lion =\
                beta2_lion * self.cache[name]['m_lion'].detach() + (1. - beta2_lion) * g
            self.cache[name]['u_lion'] = u_lion =\
                beta1_lion * self.cache[name]['m_lion'].detach() + (1. - beta1_lion) * g
            
            self.cache[name]['v_mean'] = v_mean =\
                (self.cache[name]['v'] + (t-1)*self.cache[name]['v_mean'].data)/t
            
            

            self.all_params_with_gradients.append(self.cache[name]['m'])
            self.all_params_with_gradients.append(self.cache[name]['v'])
            self.all_params_with_gradients.append(self.cache[name]['n'])
            

            m_hat = (self.cache[name]['m'] / (1. - beta1 ** float(t)))
            v_hat = ((rho*self.cache[name]['v'] + (1-rho)*v_mean.detach()) / (1. - beta2 ** float(t)))

            dparam = gamma*(m_hat + beta3 * self.cache[name]['n']) / (v_hat ** 0.5 + self.eps) + (1-gamma)*torch.sign(u_lion)
            params[name] = param.detach() - dparam*self.parameters['alpha'].detach()
            param_cache[id(param)] = params[name]

    def __str__(self):
        return 'meta / ' + str(self.optimizer)
    



class Adam(Optimizable):
    '''
    A hyperoptimizable Adam optimizer.
    '''
    def clamp_orig(x):
        return (x.tanh() + 1.) / 2.

    def unclamp_orig(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.
    
    def clamp(x,b2flag = False):
        if x>=0.99:
            x.data = 0.99*torch.ones_like(x)
        elif x<=0.5 and b2flag:
            x.data = 0.5*torch.ones_like(x)
        elif x<=0.01:
            x.data = 0.01*torch.ones_like(x)
        return x
        #return (x.tanh() + 3.) / 4.

    def unclamp(y):
        return y
    

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.95, log_eps=-6., optimizer=NoOpOptimizer()):
        self.eps = 10. ** log_eps
        parameters = {
            'alpha': torch.tensor(alpha),
            'beta1': Adam.unclamp(torch.tensor(beta1)),
            'beta2': Adam.unclamp(torch.tensor(beta2)),
        }
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        beta1 = Adam.clamp(self.parameters['beta1'])
        beta2 = Adam.clamp(self.parameters['beta2'], True)
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            self.eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }
            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + self.eps)
            #params[name].data = (param.detach() - self.parameters['alpha'] * dparam).data
            #params[name] = (param.detach() - self.parameters['alpha'] * dparam)
            params[name] = param.detach() - dparam*self.parameters['alpha'] #* dparam#.detach() #- self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adam / ' + str(self.optimizer)

class AdamBaydin(Optimizable):
    ''' Same as above, but only optimizes the learning rate, treating the
    remaining hyperparameters as constants. '''

    def __init__(
        self,
        alpha=0.01, beta1=0.9, beta2=0.99, log_eps=-8.,
        optimizer=NoOpOptimizer()
    ):
        parameters = {
            'alpha': torch.tensor(alpha),
        }
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.log_eps = log_eps
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        beta1 = self.beta1
        beta2 = self.beta2
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            10.**self.log_eps
# NOTE that we add a little `fudge factor' here because sqrt is not
# differentiable at exactly zero
                }

            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g

            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + 10. ** self.log_eps)
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adamBaydin / ' + str(self.optimizer)


class ModuleWrapper(Optimizable):
    '''
    This class tries to convert a torch.nn.Module to an Optimizable, handling
    the internal plumbing needed to update parameters correctly.
    '''
    def __init__(self, module, optimizer=NoOpOptimizer()):
        self.module = module
        parameters = {k:v for k, v in module.named_parameters(recurse=True)}
        modules = {k:v for k,v in module.named_modules()}
        if 'lm_head' in modules:
            parameters['lm_head.weight'] = self.module.lm_head.weight 
        super().__init__(parameters, optimizer)
    
    def initialize(self):
        self.optimizer.initialize()
    
    def zero_grad(self):
        """ Set all gradients to zero. """
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            if param.grad != None:
                param.grad.detach_()
                param.grad.zero_()
            else:
                param.grad = torch.zeros_like(param.data)
            #param.grad = torch.zeros_like(param.data)
        self.optimizer.zero_grad()
    
    def detach(self):
        """ Set all gradients to zero. """
        #self.module.zero_grad()
        for param in self.all_params_with_gradients:
            if param.grad != None:
                param.detach_()
            # else:
            #     param.grad = torch.zeros_like(param.data)
            #param.grad = torch.zeros_like(param.data)
        self.optimizer.zero_grad()
    
    def forward(self, *xyz):
        return self.module(*xyz)
    
    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.eval()
    
    def step(self):
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]
            
            if kk in self.parameters:
                
                m._parameters[k] = self.parameters[kk]
                # to prevent disconnection
                # m._parameters[k].data = self.parameters[kk].data
                # self.parameters[kk] = m._parameters[k]
                #
            # else:
            #     print("problem")
            #     m._parameters[k] = None

        for k, v in self.module.named_parameters(recurse=True): 
            # if k == 'lm_head.weight':
            #     a = 3
            set_param(self.module, k, v)
        #named_parameters ignore lm_head
        modules = {k:v for k,v in self.module.named_modules()}
        if 'lm_head' in modules:
            set_param(self.module, 'lm_head.weight', modules['lm_head'].weight)
            #self.module.lm_head.weight.data =self.module.transformer.wte.weight.data
            #self.module.lm_head.weight = self.module.transformer.wte.weight
            #self.parameters['lm_head.weight'] = self.module.lm_head.weight
        del modules

    def hyper_step(self):
        self.optimizer.parameters['alpha'].data = torch.tensor(0.0)
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]
            
            if kk in self.parameters:
                
                m._parameters[k] = self.parameters[kk]
                

        for k, v in self.module.named_parameters(recurse=True): 
            if k == 'lm_head.weight':
                a = 3
            set_param(self.module, k, v)
        modules = {k:v for k,v in self.module.named_modules()}
        if 'lm_head' in modules:
            # TO PRESERVE SHARED WEIGHTS
            self.module.lm_head.weight.data = self.module.transformer.wte.weight.data
            self.module.transformer.wte.weight = self.module.lm_head.weight
        del modules
