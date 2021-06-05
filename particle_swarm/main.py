import numpy as np
import matplotlib.pyplot as plt
from optimalizator.PSO import PSO



if __name__ == "__main__":
    
    function = lambda x, y: np.power(x,2) + np.power(y + 1,2) \
                            - 5*np.cos(1.5*x + 1.5) - 3*np.cos(2*x - 1.5)
    
    pso = PSO(function, 30, [-5,5], [-5,5], 0.5, 2, 2, 30)
    pso.compute()
    
    
    print("\n", pso.p_best)
    print(pso.p_best_vals)
    print(pso.particles)
    
    
    x = np.arange(-5, 5, 0.1); y = np.arange(-5, 5, 0.1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = function(xx, yy)
    
    plt.contourf(x,y,z, 20, cmap=plt.cm.ocean, origin='lower')
    plt.plot(pso.particles[0], pso.particles[1], 'rx', markersize=3)
    plt.plot(pso.p_best[0], pso.p_best[1], 'bo', markersize=3)
    plt.plot(pso.g_best[0], pso.g_best[1], 'mo', markersize=3)
    plt.show()