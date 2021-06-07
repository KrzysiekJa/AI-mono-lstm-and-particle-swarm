import math

import numpy as np
import matplotlib.pyplot as plt
from optimalizator.PSO import PSO
from scipy.special import expit

if __name__ == "__main__":
    # function = lambda x, y: np.power(x, 2) + np.power(y + 1, 2) \
    #                         - 5 * np.cos(1.5 * x + 1.5) - 3 * np.cos(2 * x - 1.5)
    # x_range = [-5, 5]
    # y_range = [-5, 5]

    # f_name = "Booth function"
    # function = lambda x, y: np.power(x + 2 * y - 7, 2) + np.power(2 * x + y - 5, 2)
    # x_range = [-10, 10]
    # y_range = [-10, 10]
    # pso = PSO(function, 109, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.5, 2, 2, 30)

    # f_name = "Eggholder function"
    # function = lambda x, y: -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
    #     np.sqrt(np.abs(x - (y + 47))))
    # x_range = [-520, 520]
    # y_range = [-520, 520]
    # pso = PSO(function, 100, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.5, 0.2, 0.2, 300)

    f_name = "Hölder table function"
    function = lambda x, y: -np.abs(
        np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.pi))))
    x_range = [-10, 10]
    y_range = [-10, 10]
    pso = PSO(function, 100, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.1, 0.01, 0.01, 300)

    step = 0.5

    # chart before opt
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = function(xx, yy)
    plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
    plt.plot(pso.particles[0], pso.particles[1], 'rx', markersize=3)
    plt.title("Before optimization")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    pso.compute()

    # print("\n", pso.p_best)
    # print(pso.p_best_vals)
    # print(pso.particles)

    # chart after opt
    plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
    #plt.plot(pso.particles[0], pso.particles[1], 'rx', markersize=3)
    plt.plot(pso.p_best[0], pso.p_best[1], 'rx', markersize=3)
    plt.plot(pso.g_best[0], pso.g_best[1], 'yo', markersize=3)
    plt.title(
        f'After optimization\nf({round(pso.g_best[0], 5)}, {round(pso.g_best[1], 5)}) = {round(function(pso.g_best[0], pso.g_best[1]), 5)}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # 3D chart
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, z, 50, cmap=plt.cm.ocean)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(25, 35)
    plt.title(f_name)
    plt.show()