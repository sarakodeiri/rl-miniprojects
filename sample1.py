import gym
import random
import numpy as np

env = gym.make('MountainCar-v0')

restart_count = 10000
randomness_down_rate = restart_count // 100
restart_steps = 5000

speed_count = 15
speed_section = 100
position_count = 57
position_section = 30
action_count = 3

alpha = 0.2
gama = 0.9
epsilon = 0.9


def calc_state(state):
    pos = state[0]
    speed = state[1]
    pos += 1.2
    pos *= position_section
    speed += 0.07
    speed *= speed_section
    return int(pos), int(speed)


def train():
    q_values = [[[2 * random.random() - 1 for i in range(action_count)] for j in range(speed_count)] for k in
                range(position_count)]
    for restarts in range(restart_count):
        state = env.reset()
        pos, speed = calc_state(state)
        randomness_rate = epsilon - ((restarts // randomness_down_rate) * 0.01)
        done = False
        while not done:
            chance = random.random()
            temp = q_values[pos][speed]
            if chance <= randomness_rate:
                next_action = random.choice([i for i in range(action_count)])
            else:
                next_action = temp.index(max(temp))
            new_state, rew, done, _ = env.step(next_action)
            new_pos, new_speed = calc_state(new_state)
            neighbors = [q_values[new_pos][new_speed][act] for act in range(action_count)]
            if done and new_pos >= 1.7 * position_section:
                q_values[pos][speed][next_action] = rew
            else:
                q_values[pos][speed][next_action] += alpha * (
                        rew + gama * (max(neighbors)) - q_values[pos][speed][next_action])
            pos, speed = new_pos, new_speed

    np.save('policies1', np.argmax(q_values, axis=2))


def play():
    policy = np.load('policies1.npy')
    state = env.reset()
    while True:
        pos, speed = calc_state(state)
        state, _, done, _ = env.step(policy[pos][speed])
        if state[0] >= 0.5:
            break
        env.render()


# train()
play()
env.close()
