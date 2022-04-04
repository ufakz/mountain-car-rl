import gym
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2

env = gym.make('MountainCar-v0')
env.reset()

goal_steps = 200
score_requirement = -198
initial_games = 10000


def model_data_preparation():
    training_data = []
    accepted_scores = []
    for game_index in range(initial_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            action = random.randrange(0, 3)
            observation, reward, done, info = env.step(action)

            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            if observation[0] > -0.2:
                reward = 1

            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1, 0]
                elif data[1] == 0:
                    output = [1, 0, 0]
                elif data[1] == 2:
                    output = [0, 0, 1]
                training_data.append([data[0], output])

        env.reset()

    print(accepted_scores)

    return training_data


training_data = model_data_preparation()


def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=adam_v2.Adam(), metrics=['accuracy'])

    return model


def train_model(training_data):
    X = np.array([i[0] for i in training_data]
                 ).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]
                 ).reshape(-1, len(training_data[0][1]))

    model = build_model(input_size=len(X[0]), output_size=len(y[0]))
    model.fit(X, y, epochs=10, validation_data=(X, y))

    print(model.history.history)

    return model


trained_model = train_model(training_data)
trained_model.save('mountain-car-model')
