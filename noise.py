import torch


class Noise():
    def __init__(self,n_actions):
        self.n_actions = n_actions

    def add_noise(self):
        noise = torch.empty(self.n_actions).normal_(mean=0,std=0.1)
        return noise
