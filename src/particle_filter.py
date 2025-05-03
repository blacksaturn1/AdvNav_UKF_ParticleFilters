import numpy as np

class ParticleFilter:
    def __init__(self, num_particles, state_dim, process_model, measurement_model, init_state_sampler):
        self.num_particles = num_particles
        self.state_dim = state_dim
        self.particles = init_state_sampler(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles
        self.process_model = process_model
        self.measurement_model = measurement_model

    def predict(self, control_input=None):
        for i in range(self.num_particles):
            self.particles[i] = self.process_model(self.particles[i], control_input)

    def update(self, measurement):
        for i in range(self.num_particles):
            self.weights[i] = self.measurement_model(self.particles[i], measurement)
        self.weights += 1.e-300  # avoid division by zero
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(
            self.num_particles, size=self.num_particles, p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)
