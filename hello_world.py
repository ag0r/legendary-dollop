import numpy as np
import neurolab as nl
import pickle
# import pdb


def main():
    file_name = 'mnist.pkl'

    input_array = np.zeros(shape=(784, 2))
    targets = np.zeros(shape=(50000, 10))

    with open(file_name, mode='r') as f:
        train_set, validation_set, test_set = pickle.load(f)

    inputs = np.array(train_set[0])
    for i in range(0, len(targets)):
        targets[i][train_set[1][i]] = 1

    for i in range(0, len(input_array)):
        input_array[i] = [0.0, 1.0]

    net = nl.net.newff(input_array, [8, 10])

    net.trainf = nl.train.train_gd
    # error = net.train(inputs, targets, epochs=100, show=1, goal=0.02)

    # pdb.set_trace()
    net.train(input=inputs, target=targets, epochs=100, show=1, goal=0.02)

    out = net.sim(inputs)

    for output, target in zip(out, targets):
        print("OUTPUT")
        print(output)
        print("TARGETS")
        print(target)


if (__name__ == "__main__"):
    main()
