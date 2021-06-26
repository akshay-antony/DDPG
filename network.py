import torch
import torch.nn as nn
from math import sqrt


class ActorNet(nn.Module):
    def __init__(self,n_states,n_actions):
        super().__init__()
        #The paper has three layers with hidden values 400 and 300
        self.fc1 = nn.Linear(n_states,400)
        self.fc2 = nn.Linear(400,300)
        self.fc3 = nn.Linear(300,n_actions)
        self.relu = nn.ReLU()
        self.out_activation = nn.Tanh()
        #self.initialize_weights() will call explicitly

    def initialize_weights(self,init_weight,f):
        nn.init.uniform_(self.fc1.weight,a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc2.weight,a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc3.weight,a = -init_weight, b = init_weight)

        nn.init.uniform_(self.fc1.bias, a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc2.bias, a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc3.bias,a = -init_weight, b = init_weight)

    def forward(self, state):
        out = self.out_activation(self.fc3(self.relu(self.fc2(self.relu(self.fc1(state))))))
        return out

    def train_step(self,optimizer):
        optimizer.zero_grad()


class CriticNet(nn.Module):
    def __init__(self,n_states,n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states,400)
        self.fc2 = nn.Linear(400 + n_actions,300)
        self.fc3 = nn.Linear(300,1)
        nn.relu = nn.ReLU()
        #no activation required

    def initialize_weights(self,init_weight,f):
        nn.init.uniform_(self.fc1.weight,a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc2.weight,a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc3.weight,a = -init_weight, b = init_weight)

        nn.init.uniform_(self.fc1.bias, a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc2.bias, a = -1/sqrt(f), b = 1/sqrt(f))
        nn.init.uniform_(self.fc3.bias,a = -init_weight, b = init_weight)


    def forward(self,state,action):
        out = self.relu(self.fc1(state))
        out = self.fc3(self.relu(self.fc2(torch.cat([out,action]))))
        return out

