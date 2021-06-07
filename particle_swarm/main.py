import numpy as np
import matplotlib.pyplot as plt
from optimalizator.PSO import PSO
from scipy.special import expit



if __name__ == "__main__":
    #function = lambda x, y: np.power(x, 2) + np.power(y + 1, 2) \
    #                         - 5 * np.cos(1.5 * x + 1.5) - 3 * np.cos(2 * x - 1.5)

    f_name   = ["Booth function", "Eggholder function", "Hölder table function"]
    function = [lambda x, y: np.power(x + 2 * y - 7, 2) + np.power(2 * x + y - 5, 2),
                lambda x, y: -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47)))),
                lambda x, y: -np.abs(np.sin(x) * np.cos(y) * expit(np.abs(1 - (np.sqrt(np.power(x, 2) + np.power(y, 2)) / np.pi))))]
                
    x_range = 10; y_range = 10; rng_step = 40
    
    
    
    for i in range(len(f_name)):
        pso = PSO(function[i], 30, [-x_range, x_range], [-y_range, y_range], 0.8, 2, 2, 100)
        
        
        # *** chart before opt ***
        x = np.linspace(-x_range, x_range, rng_step)
        y = np.linspace(-y_range, y_range, rng_step)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = function[i](xx, yy)
        
        
        fig = plt.figure()
        fig.suptitle('Before optimization: ' + f_name[i])
        fig.add_subplot(121)
        plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
        plt.plot(pso.particles[0], pso.particles[1], 'ro', markersize=3)
        plt.xlabel('x1'); plt.ylabel('x2')
        
        ax = fig.add_subplot(122, projection='3d')
        ax.contour3D(x, y, z, 50, cmap=plt.cm.ocean)
        ax.scatter(pso.particles[0], pso.particles[1], pso.fun_values, marker='x', c='r')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('z')
        plt.show()
        
        
        pso.compute()
        
        # print("\n", pso.p_best)
        # print(pso.p_best_vals)
        # print(pso.particles)
        
        
        # *** chart after opt ***
        x = np.linspace(pso.g_best[0] - x_range/2, pso.g_best[0] + x_range/2, rng_step)
        y = np.linspace(pso.g_best[1] - y_range/2, pso.g_best[1] + y_range/2, rng_step)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = function[i](xx, yy)
        
        
        fig = plt.figure()
        fig.suptitle(
            f'After optimization\nf({np.round(pso.g_best[0], 5)}, {np.round(pso.g_best[1], 5)}) = {np.round(function[i](pso.g_best[0], pso.g_best[1]), 5)}')
        fig.add_subplot(121)
        plt.contourf(x, y, z, 20, cmap=plt.cm.ocean, origin='lower')
        plt.plot(pso.p_best[0], pso.p_best[1], 'ro', markersize=3)
        plt.plot(pso.g_best[0], pso.g_best[1], 'yo', markersize=3)
        plt.xlabel('x1'); plt.ylabel('x2')
        
        ax = fig.add_subplot(122, projection='3d')
        ax.contour3D(x, y, z, 50, cmap=plt.cm.ocean)
        ax.scatter(pso.p_best[0], pso.p_best[1], pso.p_best_vals, marker='x', c='r')
        ax.scatter(pso.g_best[0], pso.g_best[1], pso.g_best_val, marker='o', c='y')
        ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('z')
        plt.show()
        
