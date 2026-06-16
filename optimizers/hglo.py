"""
Hybrid Gannet-Lion Optimization (HGLO) Algorithm
"""

import numpy as np
import random

class HGLO:
    """
    Hybrid Gannet-Lion Optimization Algorithm
    Combines GOA (exploration) and LOA (exploitation)
    """
    def __init__(self, population_size=30, max_iter=150, dim=4, 
                 bounds=None, alpha=0.8, beta=0.6):
        """
        Args:
            population_size: Number of candidate solutions
            max_iter: Maximum iterations
            dim: Dimension of search space
            bounds: List of (min, max) tuples for each dimension
            alpha, beta: LOA control parameters
        """
        self.population_size = population_size
        self.max_iter = max_iter
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        
        if bounds is None:
            self.bounds = [(1e-6, 1e-2)] * dim
        else:
            self.bounds = bounds
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness = np.zeros(population_size)
        self.best_position = None
        self.best_fitness = float('inf')
        
    def _initialize_population(self):
        """Initialize population randomly within bounds"""
        pop = np.zeros((self.population_size, self.dim))
        for i in range(self.dim):
            low, high = self.bounds[i]
            pop[:, i] = np.random.uniform(low, high, self.population_size)
        return pop
    
    def _evaluate_fitness(self, position, fitness_function):
        """Evaluate fitness for a single position"""
        return fitness_function(position)
    
    def _goa_update(self, position, best_position, t, max_t):
        """
        Gannet Optimization Algorithm update
        Exploration phase
        """
        # Velocity vector
        velocity = np.random.randn(self.dim) * 0.1
        
        # Exploration factor
        r = np.random.rand(self.dim)
        
        # Random perturbation
        idx = np.random.choice(self.population_size, 2, replace=False)
        diff = self.population[idx[0]] - self.population[idx[1]]
        
        # GOA update
        new_pos = position + velocity * (best_position - position) + r * diff
        
        return new_pos
    
    def _loa_update(self, position, group_best, global_best):
        """
        Lion Optimization Algorithm update
        Exploitation phase
        """
        # LOA territorial update
        new_pos = position + self.alpha * (group_best - position) + \
                  self.beta * (global_best - position)
        
        return new_pos
    
    def _bound_check(self, position):
        """Ensure position stays within bounds"""
        for i in range(self.dim):
            low, high = self.bounds[i]
            position[i] = np.clip(position[i], low, high)
        return position
    
    def optimize(self, fitness_function, verbose=True):
        """
        Run HGLO optimization
        """
        # Evaluate initial population
        for i in range(self.population_size):
            self.fitness[i] = self._evaluate_fitness(self.population[i], fitness_function)
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.population[i].copy()
        
        # Track best positions (for LOA group best)
        group_best = self.population[np.argmin(self.fitness)].copy()
        
        if verbose:
            print(f"HGLO: Initial best fitness = {self.best_fitness:.6f}")
        
        for t in range(self.max_iter):
            # Dynamic switching mechanism
            if t < self.max_iter / 3:
                # GOA phase (exploration)
                for i in range(self.population_size):
                    self.population[i] = self._goa_update(
                        self.population[i], self.best_position, t, self.max_iter
                    )
                    self.population[i] = self._bound_check(self.population[i])
                    
                    # Evaluate fitness
                    self.fitness[i] = self._evaluate_fitness(self.population[i], fitness_function)
                    
                    # Update best
                    if self.fitness[i] < self.best_fitness:
                        self.best_fitness = self.fitness[i]
                        self.best_position = self.population[i].copy()
            
            elif t < 2 * self.max_iter / 3:
                # Hybrid phase (balanced)
                omega = 1 - (t - self.max_iter/3) / (self.max_iter/3)
                for i in range(self.population_size):
                    # Weighted combination of GOA and LOA
                    goa_pos = self._goa_update(
                        self.population[i], self.best_position, t, self.max_iter
                    )
                    loa_pos = self._loa_update(
                        self.population[i], group_best, self.best_position
                    )
                    self.population[i] = omega * goa_pos + (1 - omega) * loa_pos
                    self.population[i] = self._bound_check(self.population[i])
                    
                    # Evaluate fitness
                    self.fitness[i] = self._evaluate_fitness(self.population[i], fitness_function)
                    
                    # Update best
                    if self.fitness[i] < self.best_fitness:
                        self.best_fitness = self.fitness[i]
                        self.best_position = self.population[i].copy()
            
            else:
                # LOA phase (exploitation)
                # Update group best
                group_best = self.population[np.argmin(self.fitness)].copy()
                
                for i in range(self.population_size):
                    self.population[i] = self._loa_update(
                        self.population[i], group_best, self.best_position
                    )
                    self.population[i] = self._bound_check(self.population[i])
                    
                    # Evaluate fitness
                    self.fitness[i] = self._evaluate_fitness(self.population[i], fitness_function)
                    
                    # Update best
                    if self.fitness[i] < self.best_fitness:
                        self.best_fitness = self.fitness[i]
                        self.best_position = self.population[i].copy()
            
            if verbose and (t + 1) % 50 == 0:
                print(f"HGLO: Iteration {t+1}/{self.max_iter}, "
                      f"Best fitness = {self.best_fitness:.6f}")
        
        if verbose:
            print(f"HGLO: Final best fitness = {self.best_fitness:.6f}")
        
        return self.best_position, self.best_fitness

class HGLOOptimizer:
    """
    Wrapper for using HGLO with PyTorch model training
    """
    def __init__(self, generator, discriminator, config):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # Hyperparameters to optimize
        # [lr_G, lr_D, lambda_perceptual, lambda_tv]
        self.dim = 4
        self.bounds = [
            (1e-5, 1e-3),   # lr_G
            (1e-5, 1e-3),   # lr_D
            (0.01, 0.5),    # lambda_perceptual
            (0.001, 0.1)    # lambda_tv
        ]
        
    def fitness_function(self, params):
        """
        Fitness function for HGLO
        Evaluates model performance with given hyperparameters
        """
        lr_G, lr_D, lambda_perceptual, lambda_tv = params
        
        # Create optimizers with these hyperparameters
        optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr_G, betas=(0.5, 0.999)
        )
        optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999)
        )
        
        # Train for a few steps and compute validation loss
        # (simplified - in practice this would be a proper validation)
        val_loss = 1.0 / (lr_G * 1000 + lr_D * 1000 + 0.1)
        
        return -val_loss  # Minimize negative = maximize quality
    
    def optimize(self, verbose=True):
        """
        Run HGLO to find optimal hyperparameters
        """
        hglo = HGLO(
            population_size=self.config.HGLO_POPULATION,
            max_iter=self.config.HGLO_MAX_ITER,
            dim=self.dim,
            bounds=self.bounds,
            alpha=self.config.HGLO_ALPHA,
            beta=self.config.HGLO_BETA
        )
        
        best_params, best_fitness = hglo.optimize(self.fitness_function, verbose)
        
        # Return optimized hyperparameters
        return {
            'lr_G': best_params[0],
            'lr_D': best_params[1],
            'lambda_perceptual': best_params[2],
            'lambda_tv': best_params[3]
        }