from Functions import is_finished


# 'O' : 1, 'X' : 2
class tictactoe():
    def __init__(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.turn = 1

    def step(self, action):
        self.state[action] = self.turn
        self.turn = self.turn % 2 + 1
        done, winner = is_finished(self.state)
        reward = 0
        if done and winner == 1: reward = 1
        if done and winner == 2: reward = -1
        if done and winner == 0: reward = 0

        return self.state, done, reward, winner

    def reset(self):
        self.state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.turn = 1

    def render(self):
        print('------------------------------------------')
        for i in range(3):
            print('|', end='')
            for j in range(3):
                if self.state[3 * i + j] == 0:
                    print(' |', end='')
                elif self.state[3 * i + j] == 1:
                    print('O|', end='')
                elif self.state[3 * i + j] == 2:
                    print('X|', end='')
            print()
        print('------------------------------------------')