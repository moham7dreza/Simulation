import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def easy_function(x):
    return 3 * (x ** 2)


def func1(x):
    return x ** 2


def func2(x):
    return (x ** 3) + 1


def func3(x):
    return (x ** 3) + (x ** 2) + 1


def hard_function(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / 2)


def main2():
    X1 = np.linspace(0, 2, 100)
    plt.plot(X1, func1(X1))
    plt.show()

    X = np.linspace(-20, 20, 1000)
    plt.plot(X, easy_function(X))
    plt.show()

    plt.plot(X, hard_function(X))
    plt.show()


def integrate(x1, x2, func=easy_function, n=100000):
    X = np.linspace(x1, x2, 1000)
    y1 = 0
    y2 = max((func(X))) + 1
    print("x1 , x2 , y1 , y2 : ", x1, x2, y1, y2)
    area = (x2 - x1) * (y2 - y1)
    print("area : ", area)
    check = []
    xs = []
    ys = []
    for i in range(n):
        x = np.random.uniform(x1, x2, 1)
        xs.append(x)
        y = np.random.uniform(y1, y2, 1)
        ys.append(y)
        if abs(y) > abs(func(x)) or y < 0:
            check.append(0)
        else:
            check.append(1)
    print("mean : ", np.mean(check))
    return np.mean(check) * area, xs, ys, check


def main1():
    print()
    print("result : ", integrate(0.3, 2.5)[0])
    print()
    print("result : ", integrate(0.3, 2.5, hard_function)[0])
    print()
    _, x, y, c = integrate(0.3, 2.5, n=100)
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['c'] = c

    X = np.linspace(0.3, 2.5, 1000)
    plt.plot(X, easy_function(X))
    plt.scatter(df[df['c'] == 0]['x'], df[df['c'] == 0]['y'], color='red')
    plt.scatter(df[df['c'] == 1]['x'], df[df['c'] == 1]['y'], color='blue')
    plt.show()


def main3(x1, x2, func=func1, n=100):
    print()
    r, x, y, c = integrate(x1, x2, func=func, n=n)
    print("result : ", round(r, 4))
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['c'] = c

    X = np.linspace(x1, x2, 1000)
    plt.plot(X, func(X))
    plt.scatter(df[df['c'] == 0]['x'], df[df['c'] == 0]['y'], color='red')
    plt.scatter(df[df['c'] == 1]['x'], df[df['c'] == 1]['y'], color='blue')
    plt.show()


def main4():
    print()
    r, x, y, c = integrate(0, 2, func=func3, n=100)
    print("result : ", round(r, 4))
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['c'] = c

    X = np.linspace(0, 2, 1000)
    plt.plot(X, func3(X))
    plt.scatter(df[df['c'] == 0]['x'], df[df['c'] == 0]['y'], color='red')
    plt.scatter(df[df['c'] == 1]['x'], df[df['c'] == 1]['y'], color='blue')
    plt.show()


if __name__ == '__main__':
    x1 = 0
    x2 = 2
    N = 1000
    print("\nAlgorithm 1 : f(x) = x*x\n")
    main3(x1, x2, func1, N)
    print("\nAlgorithm 2 : f(x) = x*x*x + 1\n")
    main3(x1, x2, func2, N)
    print("\nAlgorithm 3 : f(x) = x*x*x + x*x + 1\n")
    main3(x1, x2, func3, N)
