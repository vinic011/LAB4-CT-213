import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self, lower_bound, upper_bound):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        delta = upper_bound-lower_bound
        self.x = np.random.uniform(lower_bound,upper_bound)
        self.v = np.random.uniform(-delta,delta)
        self.current_value = None
        self.best_position = None
        self.best_value = -inf 


class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.hp = hyperparams
        self.particles = []
        for i in range(self.hp.num_particles):
            self.particles.append(Particle(lower_bound,upper_bound))
        self.best_global = Particle(lower_bound, upper_bound)
        self.count = 0

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.best_global.x

    def get_best_value(self):
        """
        Obtains the value of the best
        position so far found by the algorithm.
        :return: value of the best position.
        :rtype: float.
        """
        
        return self.best_global.best_value

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        return self.particles[self.count].x

    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        for i in range(self.hp.num_particles):
            if self.particles[i].best_value > self.best_global.best_value:
                self.best_global = self.particles[i]
        for i in range(self.hp.num_particles):
            rp = random.uniform(0.0, 1.0)
            rg = random.uniform(0.0, 1.0)
            self.particles[i].v = (self.hp.inertia_weight)*self.particles[i].v + rp*(self.hp.cognitive_parameter)*(self.particles[i].best_position-self.particles[i].x) + rg*(self.hp.social_parameter)*(self.best_global.x-self.particles[i].x)
            self.particles[i].x = self.particles[i].x + self.particles[i].v
        self.count = 0
    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        self.particles[self.count].current_value = value
        if self.particles[self.count].current_value > self.particles[self.count].best_value:
            self.particles[self.count].best_value = self.particles[self.count].current_value
            self.particles[self.count].best_position = self.particles[self.count].x
        self.count += 1
        if self.count == self.hp.num_particles:
            self.advance_generation()