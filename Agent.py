import DQN
import random
import numpy as np
from Functions import is_finished, available_actions


class AIagent_RL:
    def __init__(self, learning_rate=1e-2, restore=False, name="main"):

        self.sess = DQN.tf.Session()
        self.action_value = DQN.DQN(self.sess, learning_rate=learning_rate, name=name)
        self.sess.run(DQN.tf.global_variables_initializer())

        if restore:
            self.restore()

    def policy(self, state, turn, epsilon=0.08):
        maxvalue = -999999
        minvalue = 999999
        available = available_actions(state)
        action_list = []

        if np.random.rand(1) < epsilon:
            action_list = available
        else:
            value = self.action_value.predict(state)
            value = np.reshape(value, 9)

            if turn == 1:
                for action in available:
                    if value[action] > maxvalue:
                        action_list = []
                        maxvalue = value[action]
                        action_list.append(action)
                    elif value[action] == maxvalue:
                        action_list.append(action)
            else:
                for action in available:
                    if value[action] < minvalue:
                        action_list = []
                        minvalue = value[action]
                        action_list.append(action)
                    elif value[action] == minvalue:
                        action_list.append(action)

        return random.choice(action_list)

    def save(self):
        self.action_value.save()

    def best_save(self):
        self.action_value.best_save()

    def restore(self):
        self.action_value.restore()


class AIagent_Base:
    def policy(self, state, turn, epsilon=0):
        available = available_actions(state)
        action_list = []

        for i in available:
            state[i] = turn
            done, winner = is_finished(state)
            state[i] = 0
            if done:
                action_list.append(i)
        if len(action_list) == 0:
            action_list = available

        return random.choice(action_list)


class Human_agent:
    def policy(self, state, turn, epsilon=0):
        available = available_actions(state)

        while True:
            ret = int(input("input [0 1 2 / 3 4 5 / 6 7 8] : "))
            if ret in available:
                break
        return ret