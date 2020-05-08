import pickle


# return (done, winner)
def is_finished(state):
    for i in range(3):
        if state[3 * i] == state[3 * i + 1] and state[3 * i + 1] == state[3 * i + 2] \
                and state[3 * i] != 0:
            return True, state[3 * i]
        if state[i] == state[3 + i] and state[3 + i] == state[6 + i] and state[i] != 0:
            return True, state[i]
    if state[0] == state[4] and state[4] == state[8] and state[0] != 0:
        return True, state[0]
    if state[2] == state[4] and state[4] == state[6] and state[2] != 0:
        return True, state[2]
    for i in range(9):
        if state[i] == 0:
            return False, 0
    return True, 0


# return list
def available_actions(state):
    act = []
    for i in range(9):
        if state[i] == 0:
            act.append(i)
    return act


# save batch data
def save_batch(batch):
    f = open("./data/batch.dat", 'wb')
    pickle.dump(batch, f)
    print("Batch saved in file: ./data/batch.dat")


def best_save_batch(batch):
    f = open("./data/batch_best.dat", 'wb')
    pickle.dump(batch, f)
    print("Best batch saved in file: ./data/batch_best.dat")


# load batch data
def restore_batch():
    f = open("./data/batch.dat", 'rb')
    batch = pickle.load(f)
    print("Batch restored!")
    return batch