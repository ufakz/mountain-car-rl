import random
import gym
import keras
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
env.reset()

EPISODES = 30


model = keras.models.load_model('mountain-car-model')
print(model.summary())


def Q_Learning_NN(env, episodes=20):

    scores = []
    choices = []
    goal_steps = 200

    for i in range(episodes):
        print('Playing episode: ', i)
        score = 0
        game_memory = []
        prev_ob = []
        for step_index in range(goal_steps):
            env.render()
            if len(prev_ob) == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(
                    prev_ob.reshape(-1, len(prev_ob)))[0])

            choices.append(action)
            new_ob, reward, done, info = env.step(action)
            prev_ob = new_ob

            game_memory.append([new_ob, action])

            score += reward

            if done:
                break

        env.reset()
        scores.append(score)

    env.close()
    return scores


def random_agent(env, episodes):
    print('Training with a random agent')
    # Initialize variables to track rewards
    scores = []

    # Run for number of episodes
    for i in range(episodes):
        print('Playing episode: ', i)
        # Init params
        done = False
        tot_reward, reward = 0, 0

        while done != True:
            env.render()
            # Get random action and reward
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)

            tot_reward += reward

        env.reset()
        scores.append(tot_reward)

    env.close()

    return scores


# Run using deep neural network
Q_DNN = Q_Learning_NN(env, episodes=EPISODES)

# Run a random agent
rand_agent = random_agent(env, episodes=EPISODES)


# Plot DNN rewards/ episodes
plt.plot(np.arange(len(Q_DNN)), Q_DNN)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('dl_agent_rewards.jpg')
plt.close()

# Plot random agent rewards/ episodes
plt.plot(np.arange(len(rand_agent)), rand_agent)
plt.xlabel('Episodes')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Episodes')
plt.savefig('rand_agent_rewards.jpg')
plt.close()
