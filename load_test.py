import pickle
import numpy as np

file_name = 'mnist.pkl'

with open(file_name, mode='r') as f:
    train_set, validation_set, test_set = pickle.load(f)


# print(train_set[0][0])
# print(train_set[1][0])

inputs = np.array(train_set[0])
targets = np.array(train_set[1])

print(targets.ndim)
print(targets[1])

targets = targets.reshape((50000, 1))

print(len(targets))

print(targets.ndim)
print(targets[1])
