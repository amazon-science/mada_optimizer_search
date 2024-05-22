from math import sqrt
from matplotlib import pyplot as plt

C=3
x=0
lr = 0.1
t = 1

beta_2 = 1/(1+C**2)
v_t = 0
v_hat = 0

t_vals = []
x_vals = []

while -1 < x < 1:
    if t%3 == 0:
        grad = C 
    else:
        grad = -1

    #v_t = beta_2*v_t + (1-beta_2)*grad**2
    #v_hat = ((t-1)*v_hat + v_t)/t
    v_t = v_t + grad**2
    x = x-lr * grad/sqrt(v_t)

    t += 1
    x_vals += [x]
    if t> 100:
        break
plt.plot(x_vals)
plt.savefig('x_vals.png')
