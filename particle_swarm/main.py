import numpy as np
from optimalizator.PSO import PSO



if __name__ == "__main__":
    
    function = lambda x, y: np.power(x,2) + np.power(y + 1,2) \
                            - 5*np.cos(1.5*x + 1.5) - 3*np.cos(2*x - 1.5)
    
    pso = PSO(function, 20, [-5,5], [-5,5], 0.5, 2, 2, 100)
    pso.compute()
    
    
    print("\n", pso.p_best)
    print(pso.p_best_vals)
    print(pso.particles)
