import random
import numpy as np
import matplotlib.pyplot as plt


def probabilistic_version_of_random_walk():
    number_of_steps = 10000
    step_size = 10
    probabilities = [
        [0, 0.5, 0.5],
        [0.3, 0.7, 0],
        [0.3, 0, 0.7]
    ]
    # start_position = [5, 5]
    position_x = np.empty(number_of_steps + 1)
    position_y = np.empty(number_of_steps + 1)
    position_x[0] = np.random.randint(1, 300)
    position_y[0] = np.random.randint(1, 600)
    print("Start point is : (" + str(position_x[0]) + ", " + str(position_y[0]) + ")")
    pos_counter = 0
    for i in range(1, number_of_steps):
        x_rand_number = np.random.random()
        y_rand_number = np.random.random()
        if probabilities[0][0] < x_rand_number < probabilities[0][1]:
            position_x[i] = position_x[i - 1] + step_size
            x_rand_number = np.random.random()
            if probabilities[1][2] < x_rand_number < probabilities[1][0]:
                position_x[i + 1] = position_x[i] - step_size
            elif probabilities[1][0] < x_rand_number < 1:
                position_x[i + 1] = position_x[i]
        elif probabilities[0][2] < x_rand_number < 1:
            position_x[i] = position_x[i - 1] - step_size
            x_rand_number = np.random.random()
            if 0 < x_rand_number < 0.5:
                position_x[i + 1] = position_x[i] + step_size
            elif 0.5 < x_rand_number < 1:
                position_x[i + 1] = position_x[i]
        if 0 < y_rand_number < 0.5:
            position_y[i] = position_y[i - 1] + step_size
            y_rand_number = np.random.random()
            if 0 < y_rand_number < 0.5:
                position_y[i + 1] = position_y[i] - step_size
            elif 0.5 < y_rand_number < 1:
                position_y[i + 1] = position_y[i]
        elif 0.5 < y_rand_number < 1:
            position_y[i] = position_y[i - 1] - step_size
            y_rand_number = np.random.random()
            if 0 < y_rand_number < 0.5:
                position_y[i + 1] = position_y[i] + step_size
            elif 0.5 < y_rand_number < 1:
                position_y[i + 1] = position_y[i]
    plt.plot(position_x, position_y)
    plt.figure()
    plt.show()


if __name__ == '__main__':
    probabilistic_version_of_random_walk()
