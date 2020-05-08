import copy
from Tictactoe_Env import tictactoe
from Agent import AIagent_RL, AIagent_Base, Human_agent

env = tictactoe()
agent1 = AIagent_RL(restore=True)
agent2 = Human_agent()


def play():
    done = 0
    winner = 0
    env.reset()
    state = copy.copy(env.state)

    i = 0
    while not done:
        i += 1
        turn = copy.copy(env.turn)
        if i % 2 == 1:
            action = agent1.policy(state, turn, epsilon=0)
        else:
            action = agent2.policy(state, turn, epsilon=0)
        next_state, done, reward, winner = env.step(action)
        state = copy.copy(next_state)
        env.render()

    if winner == 0:
        print("Draw!")
    else:
        print("Winner is agent %d!" % winner)


if __name__ == "__main__":
    play()