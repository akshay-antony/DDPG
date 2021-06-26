from agent import DDPGAgent


if __name__ == '__main__':
    env_name = "Pendulum-v0"
    agent = DDPGAgent(env_name,gamma=0.99,tau=0.001)
    no_of_episodes = 100
    max_time_steps = 500 #lenth of episode
    
    for i in range(no_of_episodes):
        state = agent.env.reset()
        t = 0
        ep_rew = 0
        while True:
            action = agent.get_action(state)
            n_state, reward, done, _ = agent.env.step(action)
            transition = (state,reward,n_state,action,done)
            agent.add_memory(transition)

            if len(agent.memory)<agent.batch_size:
                continue

            transitions = agent.memory_sample()
            agent.train(transitions)
            ep_rew +=reward

            if done or t>=max_time_steps:
                print("The epoch no: {}, Episode-Reward: {}".format(i,ep_rew))
                break
            state = n_state
