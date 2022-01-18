import random
import time


def fx1(rand_num, range_len):  # x ** 2
    return rand_num * rand_num * range_len


def fx2(rand_num, range_len):  # (x ** 3) + 1
    return (rand_num * rand_num * rand_num * range_len) + (rand_num * range_len)


def fx3(rand_num, range_len):   # (x ** 3) + (x ** 2) + 1
    return (rand_num * rand_num * rand_num * range_len) + (rand_num * range_len) + (rand_num * rand_num * range_len)


def mont_carlo(N, fx_num, range_start, range_end):
    threshold = 100001
    if N < threshold:
        time_start = time.time_ns()
        proc_time_start = time.process_time_ns()
    else:
        time_start = time.time()
        proc_time_start = time.process_time()
    range_len = range_end - range_start
    sigma = 0
    for j in range(N):
        rand_num = random.uniform(range_start, range_end)
        if fx_num == 1:
            fx = fx1(rand_num, range_len)
        elif fx_num == 2:
            fx = fx2(rand_num, range_len)
        elif fx_num == 3:
            fx = fx3(rand_num, range_len)
        else:
            break
        sigma += fx
    result = round(sigma / N, 4)
    if N < threshold:
        time_end = time.time_ns()
        proc_time_end = time.process_time_ns()
    else:
        time_end = time.time()
        proc_time_end = time.process_time()
    time_parameter = " ns "
    if N > threshold:
        time_parameter = " s "
    elapsed = round(time_end - time_start, 4)
    proc_elapsed_time = round(proc_time_end - proc_time_start, 4)
    print("Result for " + str(N) + " is " + str(result)
          + " \t\tWall clock time " + str(elapsed) + time_parameter
          + " \t\tProcess time " + str(proc_elapsed_time) + time_parameter)


if __name__ == '__main__':
    print("\nAlgorithm 1 : f(x) = x*x\n")
    Mags_for = 6
    N = 100
    for i in range(Mags_for):
        mont_carlo(N, 1, 0, 2)
        N *= 10
    print("\nAlgorithm 2 : f(x) = x*x*x + 1\n")
    N = 100
    for i in range(Mags_for):
        mont_carlo(N, 2, 0, 2)
        N *= 10
    print("\nAlgorithm 3 : f(x) = x*x*x + x*x + 1\n")
    N = 100
    for i in range(Mags_for):
        mont_carlo(N, 3, 0, 2)
        N *= 10
