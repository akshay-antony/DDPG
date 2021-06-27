from agent import DDPGAgent
import torch
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("/home/akshay/PycharmProjects/RL/DDPG/runs")

if __name__ == '__main__':
    env_name = "Pendulum-v0"
    agent = DDPGAgent(env_name,gamma=0.99,tau=0.001)
    no_of_episodes = 300
    max_time_steps = 500 #lenth of episode
    
    for i in range(no_of_episodes):
        state = agent.env.reset()
        t = 0
        ep_rew = 0
        while True:
            if t%20 == 0:
                agent.env.render()
            action = agent.get_action(state)
            n_state, reward, done, _ = agent.env.step(action)
            transition = (state,reward,n_state,action,done)
            agent.add_memory(transition)

            if len(agent.memory.replay_memory)<agent.batch_size:
                continue

            transitions = agent.memory_sample()
            agent.train(transitions)
            ep_rew +=reward

            if done or t>=max_time_steps:
                print("The epoch no: {}, Episode-Reward: {}".format(i,ep_rew))
                writer.add_scalar(tag='episode_rewards',scalar_value=ep_rew,global_step=i+1)
                break
            state = n_state

    path1 = "/home/akshay/PycharmProjects/RL/DDPG/actor.pth"
    path2 = "/home/akshay/PycharmProjects/RL/DDPG/critic.pth"
    path3 = "/home/akshay/PycharmProjects/RL/DDPG/target_actor.pth"
    path4 = "/home/akshay/PycharmProjects/RL/DDPG/target_critic.pth"

    torch.save(agent.actor.state_dict(),path1)
    torch.save(agent.critic.state_dict(),path2)
    torch.save(agent.target_actor.state_dict(),path3)
    torch.save(agent.target_critic.state_dict(),path4)
    writer.close()
