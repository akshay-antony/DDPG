import torch
import torch.nn as nn
import torch.optim as optim
import gym
from network import ActorNet, CriticNet
from replaymemory import ReplayMemory
from noise import Noise
import random


class DDPGAgent:
    def __init__(self,env_name,gamma,tau):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        self.actor = ActorNet(self.n_states,self.n_actions)
        self.actor.initialize_weights(3e-3,400)
        self.target_actor = ActorNet(self.n_states,self.n_actions)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        self.critic = CriticNet(self.n_states,self.n_actions)
        self.critic.initialize_weights(3e-3,400)
        self.target_critic = CriticNet(self.n_states,self.n_actions)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(),self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),self.critic_lr)

        self.replay_memory_size = 1e6
        self.batch_size = 64
        self.memory = ReplayMemory(self.replay_memory_size)

        self.noise = Noise(self.n_actions)

    def get_action(self,state):
        action = self.actor.forward(state)  #check if unsqueeze req
        action += self.noise.add_noise()
        return (torch.clamp(action,min=-1,max=1)).numpy()

    def add_memory(self,transition):
        self.memory.replay_memory.append(transition)

    def memory_sample(self):
        return random.sample(self.memory.replay_memory,self.batch_size)

    def train(self,transitions):
        states = [transition[0] for transition in transitions]
        rewards = [transition[1] for transition in transitions]
        n_states = [transition[2] for transition in transitions]
        actions = [transition[3] for transition in transitions]
        done = [transition[4] for transition in transitions]

        states = torch.as_tensor(states,dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        n_states = torch.as_tensor(n_states,dtype=torch.float32)
        actions = torch.as_tensor(actions,dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)

        #train the critic
        n_actions = self.target_actor(states)
        targets = self.calculate_return(rewards,n_states,done,n_actions)
        critic_values = self.critic.forward(states,actions)
        critic_loss = nn.MSELoss(targets,critic_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #train the actor
        actor_new_actions = self.actor(states)
        actor_loss = -self.critic(states,actor_new_actions)
        actor_loss = actor_loss.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_step(self.actor,self.target_actor)
        self.update_step(self.critic,self.target_critic)

    def calculate_return(self,rewards,n_states,done,actions):
        out = self.target_critic.forward(n_states,actions)
        targets = rewards + self.gamma*out*(1-done)
        return targets

    def update_step(self,model,target_model):
        for model_params, target_model_params in zip(model.parameters(),target_model.parameters()):
            target_model_params.copy((1-self.tau)*target_model_params+self.tau*model_params)
