# 灵活使用clf, draw, pause实现动图; plt.show则是定格图案
import numpy as np
import matplotlib.pyplot as plt

# 保存gif的工具
import imageio
image_list = []

def func(x,y):
    res = np.exp(-(x+1)**2-(y+0.5)**2)-np.exp(-(x-1)**2-(y-0.5)**2)
    return res*2

def dfunc(x,y):
    dx = -2*(x+1)*np.exp(-(x+1)**2-(y+0.5)**2) + 2*(x-1)*np.exp(-(x-1)**2-(y-0.5)**2)
    dy = -2*(y+0.5)*np.exp(-(x+1)**2-(y+0.5)**2) + 2*(y-0.5)*np.exp(-(x-1)**2-(y-0.5)**2)
    return dx*2,dy*2

def gradient_descent(x_start, y_start, epochs, learning_rate):
    theta_x = []
    theta_y = []
    temp_x = x_start
    temp_y = y_start

    theta_x.append(temp_x)
    theta_y.append(temp_y)

    for i in range(epochs):
        dx,dy = dfunc(temp_x, temp_y)
        temp_x = temp_x - dx*learning_rate
        temp_y = temp_y - dy*learning_rate

        theta_x.append(temp_x)
        theta_y.append(temp_y)

    return theta_x, theta_y

#------------------------------------------------------------
def mat_plot(epochs=10, learning_rate=0.3):
    lr = learning_rate
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X,Y = np.meshgrid(x,y)
    Z = func(X, Y)

    dx,dy = gradient_descent(-0.9, -0.25, epochs, lr)

    plt.clf()
    plt.scatter(dx, dy, color='r')
    plt.plot(dx, dy, linewidth=1, linestyle='--', label='lr=%4.2f'%lr)

    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=0.5, fontsize=6)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis([-3, 3, -2, 2])
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(.001)


for i in np.linspace(0.01, 1.2, 50):
    mat_plot(10, i)
    plt.savefig('./figures/SGD.png')
    if i > 0.01:
        image_list.append(imageio.imread('./figures/SGD.png'))
imageio.mimsave('./figures/SGD.gif', image_list, duration=0.1)
plt.show() # 使图片定格


#------------------------------------------------------------
# a 3d version
image_list = []
def mat_plot(epochs=10, learning_rate=0.3):
    lr = learning_rate
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-3.0, 3.0, delta)
    X,Y = np.meshgrid(x,y)
    Z = func(X, Y)

    dx,dy = gradient_descent(-0.9, -0.25, epochs, lr)

    plt.clf()
    ax = plt.axes(projection='3d')
#    ax.plot_surface(X,Y,Z, cmap='coolwarm')
    ax.contour(X,Y,Z)

    ax.scatter(dx, dy, func(np.array(dx),np.array(dy)), color='r')
    ax.plot(dx, dy, func(np.array(dx), np.array(dy)), linewidth=1, linestyle='--', label='lr=%4.2f'%lr)

#    CS = plt.contour(X, Y, Z)
#    plt.clabel(CS, inline=0.5, fontsize=6)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis([-3, 3, -2, 2])
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.draw()
    plt.pause(.1)

#------------------------------------------------------------


for i in np.linspace(0.01, 1.2, 100):
    mat_plot(20, i)
    plt.savefig('./figures/SGD_3d.png')
    if i > 0.01:
        image_list.append(imageio.imread('./figures/SGD_3d.png'))
imageio.mimsave('./figures/SGD_3d.gif', image_list, duration=0.1)
plt.show()