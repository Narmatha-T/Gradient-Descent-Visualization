import numpy as np
import matplotlib.pyplot as plt

# 保存gif的工具
import imageio
image_list = []

# dg为gradient
class GD():
    def __init__(self, eta=1e-3):
        self.eta = eta

    def Delta(self, dg):
        return -self.eta * dg


class Adagrad():
    def __init__(self, eta=1e-3, epsilon=1e-8):
        self.eta = eta
        self.epsilon = epsilon
        self.past_g = 0

    def Delta(self, dg):
        self.past_g += dg**2
        inv_G = np.diag((self.past_g**(1/2)+self.epsilon)**(-1))
        return -self.eta * inv_G @ dg

class RMSprop():
    def __init__(self, eta=1e-3, gamma = 0.9, epsilon=1e-8):
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.v = 0

    def Delta(self, dg):
        self.v = self.gamma * self.v + (1-self.gamma)*dg*dg
        inv_V = np.diag((self.v**(1/2)+self.epsilon)**(-1))
        return -self.eta * inv_V @ dg


class Momentum():
    def __init__(self, eta=1e-3, gamma = 0.9):
        self.eta = eta
        self.gamma = gamma
        self.m = 0

    def Delta(self, dg):
        self.m = self.gamma*self.m - (1-self.gamma)*dg
        return self.eta * self.m

class Adam():
    def __init__(self, eta=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m, self.v = 0,0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
        self.t = 0

    def Delta(self, dg):
        self.t += 1
        self.m = self.beta1*self.m - (1-self.beta1)*dg
        self.v = self.beta2*self.v + (1-self.beta2)*dg**2
        hat_m = self.m/(1-self.beta1**self.t)
        hat_v = self.v/(1-self.beta2**self.t)
        inv_hat_V = np.diag((hat_v**(1/2) + self.epsilon)**(-1))
        return self.eta * inv_hat_V @ hat_m


Beale = lambda x, y: (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2

def Beale_Gradient(theta):
    x, y = theta
    dx = -12.75 + 3*y + 4.5*y**2 + 5.25*y**3 + 2*x*(3-2*y-y**2-2*y**3+y**4+y**6)
    dy = 6*x*(0.5+1.5*y+2.625*y**2+x*(-1/3-1/3*y-y**2+2/3*y**3+y**5))
    return np.array([dx, dy])

def Beale_Gradient_Descent(theta_t, optimizer, max_epochs=1e5, tol=1e-5):
    paths = [theta_t]
    for epoch in range(int(max_epochs)):
        dg = Beale_Gradient(theta_t)
        theta_new = theta_t+optimizer.Delta(dg)
        paths.append(theta_new)
        if sum((theta_new-theta_t)**2) < tol**2:
            return (np.squeeze(paths))
        theta_t = theta_new
    return (np.squeeze(paths))

start = np.array([1,1])
start_ = start.reshape(-1,1)

GD_path = Beale_Gradient_Descent(start, GD(eta=1e-2))
Adagrad_path = Beale_Gradient_Descent(start, Adagrad(eta=1))
RMSprop_path = Beale_Gradient_Descent(start, RMSprop(eta=5e-2))
Momentum_path = Beale_Gradient_Descent(start, Momentum(eta=1e-1))
Adam_path = Beale_Gradient_Descent(start, Adam(eta=1))

# print({'GD': len(GD_path)})
# print({'Adagrad': len(Adagrad_path)})
# print({'RMSprop': len(RMSprop_path)})
# print({'Momentum': len(Momentum_path)})
# print({'Adam': len(Adam_path)})

# print({"nstep": len(RMSprop_path), "eta": 1e-2, "RMSprop": RMSprop_path[-1]})


plt.figure(figsize=(10,6))

def mat_plot(step):

    plt.clf()

    # ax = plt.axes(projection='3d')

    path = GD_path
    x_path, y_path = path[:,0][:step], path[:,1][:step]
    plt.plot(x_path, y_path, label='GD')

    path = Adagrad_path
    x_path, y_path = path[:,0][:step], path[:,1][:step]
    plt.plot(x_path, y_path, label='Adagrad', linestyle='-')

    path = RMSprop_path
    x_path, y_path = path[:,0][:step], path[:,1][:step]
    plt.plot(x_path, y_path, label='RMSprop', linestyle='--')

    path = Momentum_path
    x_path, y_path = path[:,0][:step], path[:,1][:step]
    plt.plot(x_path, y_path, label='Momentum')

    path = Adam_path
    x_path, y_path = path[:,0][:step], path[:,1][:step]
    plt.plot(x_path, y_path, label='Adam', linestyle='-.')
    # plt.xlim((-1,4))
    # plt.ylim((-1,3))


    plt.title('Gradient Descent')
    x1_con = np.linspace(-1,3.5,100)
    x2_con = np.linspace(-1.5,1.5,100)
    x1_mes, x2_mes = np.meshgrid(x1_con,x2_con)
    y_con = Beale(x1_mes,x2_mes)
    # plt.plot(x1_mes, x2_mes)
    cset = plt.contourf(x1_mes,x2_mes,y_con, levels=20, alpha=.75, cmap='hot_r')
    plt.colorbar(cset)
    C = plt.contour(x1_mes,x2_mes,y_con, levels=20) #contour依次传入x1，x2，y就可以产生等值云图
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(1, 1, marker="8", color="b", s=40)
    plt.scatter(3, 0.5, marker="*", color="brown", s=100)
    plt.legend(loc='upper left')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.3)

for step in np.arange(1, 100, 2):
    mat_plot(step)
    plt.savefig('./figures/GD_visualization.png')
    image_list.append(imageio.imread('./figures/GD_visualization.png'))
imageio.mimsave('./figures/GD_visualization.gif', image_list, duration=0.1)
plt.show()