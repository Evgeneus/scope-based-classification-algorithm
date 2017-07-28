from scipy.special import binom
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random

from scipy.optimize import fmin_tnc
import math
from classifier_utils import find_jt
import pandas as pd

cost = 5
J = 30
Nt=5
# z=0.3
# Zs=(z*0.5**Nt)/(z*0.5**Nt + (2.*(1-z)/(Nt+1))*(1-1./(2**Nt+1)))
# alpha_new=6.
# beta_new=1.
# a_tw_avg = 0.5+0.5*(alpha_new/(alpha_new+beta_new))
# acc = Zs*0.5 + (1-Zs)*a_tw_avg
acc=0.86


def fun(theta, Jt):
    p_fe = 0.
    for k in range(Jt, J+1, 1):
        p_fe += binom(J, k)*(1-acc)**k*acc**(J-k)
    p_fe *= theta
    p_fi = 0.
    for k in range(J-Jt+1, J+1, 1):
        p_fi += binom(J, k)*(1-acc)**k*acc**(J-k)
    p_fi *= (1-theta)
    z = p_fe * cost + p_fi
    return z


def plot_loss():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0.0, 1.1, 0.01)
    y = np.arange(16, 31, 1)
    # x = y = np.arange(-3.0, 3.0, 0.05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    Gx, Gy = np.gradient(Z)  # gradients with respect to x and y
    G = (Gx ** 2 + Gy ** 2) ** .5  # gradient magnitude
    N = G / G.max()  # normalize 0..1
    ax.plot_surface(
        X, Y, Z, rstride=1, cstride=1,
        facecolors=cm.jet(N),
        linewidth=0, antialiased=False, shade=False)
    # ax.scatter(X, Y, Z)
    # ax.plot_wireframe(X, Y, Z,)


    ax.set_xlabel('Theta')
    ax.set_ylabel('Jt')
    ax.set_zlabel('Loss')
    ax.text2D(0.05, 0.95, "Loss vs Theta vs Jt", transform=ax.transAxes)

    plt.show()


def loss_vs_tests():
    z = 0.22
    theta = 0.1
    cost = 10
    data = []
    papers_page = 10
    for Nt in range(1, 11, 1):
        for J in [2, 3, 5, 10]:
            Zs = (z * 0.5 ** Nt) / (z * 0.5 ** Nt + (2. * (1 - z) / (Nt + 1)) * (1 - 1. / (2 ** Nt + 1)))
            acc_tw_avg = 2**(Nt+1)*(Nt+1)*(1-0.5**(Nt+2))/((2**(Nt+1)-1)*(Nt+2))
            acc_avg = Zs * 0.5 + (1 - Zs) * acc_tw_avg

            Jt_opt, loss = find_jt(theta, J, acc_avg, cost)
            budget = J*(Nt+papers_page)/float(papers_page)
            data.append([Nt, J, Jt_opt, loss, budget, cost])

    with open('../data/loss_tests_formula_in95.csv', 'a') as f:
        pd.DataFrame(data, columns=['Nt', 'J', 'Jt_opt', 'loss', 'budget', 'cost']).\
            to_csv('../data/loss_tests_formula_11111.csv', index=False, header=False)



# z=0.3
# Zs=(z*0.5**Nt)/(z*0.5**Nt + (2.*(1-z)/(Nt+1))*(1-1./(2**Nt+1)))
# alpha_new=6.
# beta_new=1.
# a_tw_avg = 0.5+0.5*(alpha_new/(alpha_new+beta_new))
# acc = Zs*0.5 + (1-Zs)*a_tw_avg

if __name__ == '__main__':
    loss_vs_tests()
    # plot_loss()
    # find_min()
    # for Jt in [3, 4, 5]:
    #     print Jt
    #     print fun(0.5, Jt)
