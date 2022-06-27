import gym
#from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


env = gym.make('MountainCar-v0')
env.reset()

# hard code observation space size
DISC_OS_SIZE = [20] * len(env.observation_space.high)

disc_os_win_size = (env.observation_space.high - env.observation_space.low) / DISC_OS_SIZE

# initialize q-table with random numbers
q_table = np.random.uniform(low=-2, high=0, size=(DISC_OS_SIZE + [env.action_space.n]))

# set learning rate
LEARNING_RATE = 0.01
# set discount rate
DISCOUNT = 0.95
# set episode number
EPISODES = 25000
# variable to track current episode
FOR_EVERY = 2000

# set epsilon to control randomness
epsilon = 0.8
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAY - START_EPSILON_DECAY)

# track rewards 
ep_rewards= []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# helper function to convert state from continuous to discrete
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / disc_os_win_size
    return tuple(discrete_state.astype(np.int64)) 

done = False
# train agent
for episode in range(EPISODES): 
    discrete_state = get_discrete_state(env.reset()) # initialize discrete state
    episode_reward = 0   # initialize reward for each episode
    # print every 2000th episode 
    if episode % FOR_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
        
    # loop until the flag is reached
    done=False 
    while not done:
        
        if np.random.random() > epsilon:
            # get action from q-table
            action = np.argmax(q_table[discrete_state]) 
        else:
            # get random action
            action = np.random.randint(0, env.action_space.n)
        # take action
        new_state, reward, done, _ = env.step(action)
        # update reward 
        episode_reward += reward
        # convert state to discrete
        new_discrete_state = get_discrete_state(new_state)

        # render environment when running from console
        if render:
            env.render()
            '''
            #render environment when running in jupyter 
            env_screen = env.render(mode='rgb_array')
            plt.imshow(env_screen)
            '''
        if not done:
            # get maximum possible q value for new state
            max_future_q = np.max(q_table[new_discrete_state])
            # get current q value
            current_q = q_table[discrete_state + (action, )]
            # get new q value 
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # update q table with new q value
            q_table[discrete_state + (action, )] = new_q
        # if done, grant reward  
        elif new_state[0] >= env.goal_position:
            #print(f'We made it on episode {episode}')
            q_table[discrete_state + (action, )] = 0
       
        # update discrete state for next iteration
        discrete_state = new_discrete_state
       
    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY: 
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    
    # update episode rewards dictionary every iteration
    if not episode % FOR_EVERY == 0:
        avg_reward = sum(ep_rewards[-FOR_EVERY:]) / len(ep_rewards[-FOR_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-FOR_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-FOR_EVERY:]))
     
    if episode % FOR_EVERY == 0:
        avg_reward = sum(ep_rewards[-FOR_EVERY:]) / len(ep_rewards[-FOR_EVERY:])
        print(f'ep: {episode}, avg: {avg_reward}, min: {min(ep_rewards[-FOR_EVERY:])}, max: {max(ep_rewards[-FOR_EVERY:])}')
        
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.savefig('aggr rewards.pdf')
plt.show()

