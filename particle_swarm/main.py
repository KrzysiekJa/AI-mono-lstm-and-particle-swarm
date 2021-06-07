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
    # pso = PSO(function, 30, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.5, 2, 2, 30)

    # f_name = "Goldstein–Price function"
    # function = lambda x, y: (1 + np.power(x + y + 1, 2) * (
    #             19 - 14 * x + 3 * np.power(x, 2) - 14 * y + 6 * x * y + 3 * np.power(y, 2))) \
    #                         * (30 + np.power(2 * x - 3 * y, 2) * (
    #             18 - 32 * x + 12 * np.power(x, 2) + 48 * y - 36 * x * y + 27 * np.power(y, 2)))
    # x_range = [-5, 5]
    # y_range = [-5, 5]
    # pso = PSO(function, 30, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.5, 2, 2, 30)

    f_name = "Eggholder function"
    function = lambda x, y: -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(
        np.sqrt(np.abs(x - (y + 47))))
    x_range = [-520, 520]
    y_range = [-520, 520]
    pso = PSO(function, 30, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.5, 0.2, 0.2, 30)

    # f_name = "Hölder table function"
    # function = lambda x, y: -np.abs(
    #     np.sin(x) * np.cos(y) * expit(np.abs(1 - (np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.pi))))
    # x_range = [-5, 5]
    # y_range = [-5, 5]
    # pso = PSO(function, 30, [x_range[0], x_range[1]], [y_range[0], y_range[1]], 0.1, 0.01, 0.01, 3000)


    step = 0.5
    x = np.arange(x_range[0], x_range[1], step)
    y = np.arange(y_range[0], y_range[1], step)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = function(xx, yy)
    plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
    plt.plot(pso.particles[0], pso.particles[1], 'rx', markersize=3)
    plt.title("Before optimization")
    plt.show()

    pso.compute()

    # print("\n", pso.p_best)
    # print(pso.p_best_vals)
    # print(pso.particles)

    plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
    plt.plot(pso.particles[0], pso.particles[1], 'rx', markersize=3)
    plt.plot(pso.p_best[0], pso.p_best[1], 'bo', markersize=3)
    plt.plot(pso.g_best[0], pso.g_best[1], 'yo', markersize=3)
    plt.title(
        f'After optimization\nf({round(pso.g_best[0], 5)}, {round(pso.g_best[1], 5)}) = {round(function(pso.g_best[0], pso.g_best[1]), 5)}')
    plt.show()

    plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
    plt.title(f_name)
    plt.show()