import numpy as np



class PSO:
    
    def __init__(self, fitFunction, N=50, xrng=[-1,1], yrng=[-1,1], \
                    w=0.8, c_1=2, c_2=2, max_iter=100):
        self.fitFunction = fitFunction
        self.N   = N
        self.w   = w if np.abs(w) < 1.0 else 0.9999
        self.c_1 = c_1
        self.c_2 = c_2
        self.max_iter = max_iter
        
        self.particles = np.random.rand(2, N) # for x and y coordinates
        self.particles[0,:] = self.particles[0,:] * (xrng[1]-xrng[0]) + xrng[0]
        self.particles[1,:] = self.particles[1,:] * (yrng[1]-yrng[0]) + yrng[0]
        self.velocities = np.random.rand(2, N) # for x and y coordinates
        # vectors generating with cut leght
        self.velocities[0,:] = 0.2 *(self.velocities[0,:] * (xrng[1]-xrng[0]) + xrng[0])
        self.velocities[1,:] = 0.2 *(self.velocities[1,:] * (yrng[1]-yrng[0]) + yrng[0])
        
        self.fun_values = np.array(list(map(self.fitFunction, \
                                    self.particles[0,:], self.particles[1,:])))
        self.p_best = self.particles
        self.p_best_vals = self.fun_values
        self.g_best = self.particles[:, int(np.argmin(self.fun_values))]
        self.g_best_val = np.min(self.fun_values)
        
        self.iter = 1
        '''
        print(self.particles)
        print(self.velocities)
        print(self.fun_values)
        print(self.p_best)
        print(self.p_best_vals)
        print(self.g_best)
        print(self.g_best_val)
        '''
    
    
    
    def compute(self):
        
        while self.iter < self.max_iter:
            self.iter += 1
            
            # calculating new positions of particles
            new_velocities = self.w * self.velocities
            r_1 = np.random.rand(1, self.N)
            r_2 = np.random.rand(1, self.N)
            
            new_velocities += self.c_1 * r_1 * (self.p_best - self.particles)
            new_velocities += self.c_2 * r_2 * \
                            (np.ones((2, self.N)) * self.g_best.reshape((2,1)) - self.particles)
            
            
            # stop condition
            if np.sum(self.velocities - new_velocities) == 0:
                break
            
            self.velocities = new_velocities
            self.particles  = self.particles + self.velocities
            
            
            # calculatations of function values for new positions of particles
            self.fun_values = np.array(list(map(self.fitFunction, \
                                        self.particles[0,:], self.particles[1,:])))
            
            # updates of minima positions and values
            self.p_best[:,self.fun_values < self.p_best_vals] = self.particles[:,self.fun_values < self.p_best_vals]
            self.p_best_vals = np.minimum(self.p_best_vals, self.fun_values)
            
            new_g_best_val = np.min(np.minimum(self.g_best_val, self.fun_values))
            
            if self.g_best_val != new_g_best_val:
                self.g_best = self.particles[:, int(np.argmin(self.fun_values))]
            
            self.g_best_val = new_g_best_val
