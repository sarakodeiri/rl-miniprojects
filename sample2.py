import gym
import random
import numpy as np

env = gym.make('Pendulum-v0')

action_section = 17
degree_section = 9
speed_section = 1.5
degree_count = 40
speed_count = 25
restart_count = 8000
depth_count = 200

alpha = 0.24
gama = 0.96
randomness_down_rate = 500


def calc_state(state):
    theta = np.arctan([state[1] / state[0]])[0]
    speed = state[2]
    theta = theta - np.pi if (theta >= 0 and state[1] <= 0) else (
        theta + np.pi if (theta <= 0 and state[1] >= 0) else theta)
    theta = theta * 180 / np.pi
    return [int((theta + 180) // degree_section), int((speed + 8) * speed_section)]


def train():
    q_values = [[[((k - action_section // 2) * (i - degree_section // 2)) / 160 for k in range(action_section)]
                 for j in range(speed_count)] for i in range(degree_count)]
    for random_restarts in range(restart_count):
        state = env.reset()
        state_pars = calc_state(state)
        k, j = state_pars[0], state_pars[1]
        randomness_prob = 0.9 - ((random_restarts // randomness_down_rate) * 0.01)
        for i in range(depth_count):
            if random.random() <= randomness_prob:
                a = random.randint(0, action_section - 1)
            else:
                a = q_values[k][j].index(max(q_values[k][j]))
            new_state, rew, _, _ = env.step([(a - (action_section // 2)) / (action_section // 4)])
            new_state = calc_state(new_state)
            neighbors = [(q_values[new_state[0]][new_state[1]][act]) for act in range(action_section)]
            q_values[k][j][a] += alpha * (rew + gama * (max(neighbors)) - q_values[k][j][a])
            k = new_state[0]
            j = new_state[1]

    temp_policy = np.argmax(q_values, axis=2)
    policy = [[(i - (action_section // 2)) / (action_section // 4) for i in q] for q in temp_policy]
    np.save('policies2', policy)


def play():
    policy = np.load('policies2.npy')
    state = env.reset()
    while True:
        state_mode = calc_state(state)
        state, _, _, _ = env.step([policy[state_mode[0]][state_mode[1]]])
        env.render()


# train()
play()
env.close()
