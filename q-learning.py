import gym
#from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


env = gym.make('MountainCar-v0')
#env.action_space = spaces.Box(low=-1, high = 1, shape = (3,), dtype='int64')
env.reset()

# hard code observation space size
DISC_OS_SIZE = [20] * len(env.observation_space.high)

disc_os_win_size = (env.observation_space.high - env.observation_space.low) / DISC_OS_SIZE

# initialize q-table with random numbers
q_table = np.random.uniform(low=-2, high=0, size=(DISC_OS_SIZE + [env.action_space.n]))
done = False

# set learning rate
LEARNING_RATE = 0.1
# set discount rate
DISCOUNT = 0.95
# set episode number
EPISODES = 50000
# variable to track current episode
FOR_EVERY = 2000

# set epsilon to control randomness
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# track rewards 
ep_rewards= []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# helper function to convert state from continuous to discrete
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / disc_os_win_size
    return tuple(discrete_state.astype(np.int64)) 

# initialize discrete state
discrete_state = get_discrete_state(env.reset())

for episode in range(EPISODES): 
    episode_reward = 0
    if episode % FOR_EVERY == 0:
        print(episode)
    while not done:
        # select action that optimizes q_value
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        # render environment
        
        env_screen = env.render(mode='rgb_array')
        sleep(0.03)
        plt.imshow(env_screen)

    
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        # if done, grant reward  
        elif new_state[0] >= env.goal_position:
            print(f'We made it on episode {episode}')
            q_table[discrete_state + (action, )] = 0
        # update discrete state for next iteration
        discrete_state = new_discrete_state
        
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING: 
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    
    if not episode % FOR_EVERY == 0:
        avg_reward = sum(ep_rewards[-FOR_EVERY:]) / len(ep_rewards[-FOR_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(avg_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-FOR_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-FOR_EVERY:]))
        
        #print(f'ep: {episode}, avg: {avg_reward}, min: {min(ep_rewards[-FOR_EVERY:])}, max: {max(ep_rewards[-FOR_EVERY:])}')
        
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label='avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label='min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label='max')
plt.legend(loc=4)
plt.show()

