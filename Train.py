import random
import copy
import numpy as np
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base
from collections import deque
from Functions import save_batch, restore_batch, best_save_batch

restore = False

input_size = 9
output_size = 9
max_batch_size = 1000000

learning_rate = 0.001
discount_factor = 0.9
epsilon = 0.1

train_episode = 1000
verify_episode = 10000

env = tictactoe()
agent = AIagent_RL(learning_rate=learning_rate, restore=restore)
agent_base = AIagent_Base()


def update(agent, batch, dis=discount_factor):
    x_stack = np.empty(0).reshape(0, input_size)
    y_stack = np.empty(0).reshape(0, output_size)

    for state, action, reward, next_state, next_turn, done in batch:

        action_value = agent.action_value.predict(state)
        action_value = np.reshape(action_value, 9)

        if not done:
            next_action = agent.policy(next_state, next_turn, epsilon=0)
            next_action_value = agent.action_value.predict(next_state)
            next_action_value = np.reshape(next_action_value, 9)
            action_value[action] = reward + dis * next_action_value[next_action]
        else:
            action_value[action] = reward

        _action_value = copy.deepcopy(action_value)
        _state = copy.deepcopy(state)

        y_stack = np.vstack([y_stack, _action_value])
        x_stack = np.vstack([x_stack, _state])

    agent.action_value.update(x_stack, y_stack)


def train():
    if restore:
        batch = restore_batch()
    else:
        batch = deque()

    best_win_rate = 0.95
    episode = 0
    idx = 0
    while True:  # episode < total_episode:
        idx += 1

        # training stage (self-training)
        for _ in range(train_episode):
            episode += 1
            done = 0
            env.reset()
            state = copy.copy(env.state)

            while not done:
                turn = copy.copy(env.turn)
                action = agent.policy(state, turn, epsilon=epsilon)
                next_state, done, reward, winner = env.step(action)

                data = copy.deepcopy((state, action, reward, next_state, turn % 2 + 1, done))
                batch.append(data)
                if len(batch) > max_batch_size:
                    batch.popleft()

                state = copy.copy(next_state)

        # update
        for _ in range(20):
            mini_batch = random.sample(batch, 1000)
            update(agent, mini_batch)

        # verification stage
        win = lose = draw = 0
        for i in range(verify_episode):
            done = 0
            env.reset()
            state = copy.copy(env.state)

            j = 0
            while not done:
                j += 1
                turn = copy.copy(env.turn)
                if (i + j) % 2 == 1:
                    # epsilon 0
                    action = agent.policy(state, turn, epsilon=0)
                else:
                    action = agent_base.policy(state, turn, epsilon=0)
                next_state, done, reward, winner = env.step(action)
                state = copy.copy(next_state)

            if winner == 0:
                draw += 1
            elif (i + j) % 2 == 1:
                win += 1
            else:
                lose += 1
        win_rate = (win + draw) / verify_episode
        print("[Episode %d] Win : %d Draw : %d Lose : %d Win_rate: %.4f" % (episode, win, draw, lose, win_rate))

        # save data
        agent.save()
        if idx % 10 == 0:
            save_batch(batch)

        if win_rate >= best_win_rate:
            best_win_rate = win_rate
            agent.best_save()
            best_save_batch(batch)


if __name__ == "__main__":
    train()