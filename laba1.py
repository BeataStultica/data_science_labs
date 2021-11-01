import numpy as np
import math as mt
import matplotlib.pyplot as plt


def stats(data):
    dS = np.var(data)
    return {"median": np.median(data), "ds": dS, "scvS": mt.sqrt(dS)}


def stats_print(stats):
    for name, value in stats.items():
        print(f"{name}: {value}")


def exponential(target=0.5, size=1000):
    beta = 1.0/target
    exp = np.random.exponential(beta, size)
    return exp - np.median(exp)


def normal(n=1000, dm=0, dsig=5):
    return ((np.random.randn(n))*dsig)+dm


def linear_model(error, n=1000):
    S0 = np.zeros((n))
    S1 = np.zeros((n))
    for i in range(n):
        S0[i] = (10+0.05*i)
        S1[i] = S0[i]+error[i]
    return {"s0": S0, "s1": S1}


def quad_model(error, n=1000):
    S0 = np.zeros((n))
    S1 = np.zeros((n))
    for i in range(n):
        S0[i] = (10+0.00009*i*i)
        S1[i] = S0[i]+error[i]
    return {"s0": S0, "s1": S1}


def plot(error, S0, S1):
    plt.hist(error, bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    plt.plot(S1)
    plt.plot(S0)
    plt.ylabel('динаміка продажів')
    plt.show()
    plt.hist(error, bins=20, alpha=0.5, label='S')
    plt.hist(S0, bins=20, alpha=0.5, label='S1')
    plt.hist(S1, bins=20, alpha=0.5, label='S3')
    plt.show()


def main():
    exp = exponential()
    exp_stats = stats(exp)
    norm = normal()
    norm_stats = stats(norm)
    print("normal model stats: ")
    stats_print(norm_stats)
    print("\nexponentail model stats: ")
    stats_print(exp_stats)

    quad_exp = quad_model(exp)
    quad_exp_stats = stats(quad_exp["s1"])
    print("\n Quad model with exp errors stats: ")
    stats_print(quad_exp_stats)

    quad_norm = quad_model(norm)
    quad_norm_stats = stats(quad_norm["s1"])
    print("\n Quad model with norm errors stats: ")
    stats_print(quad_norm_stats)

    quad_exp_add = plot(exp, quad_exp["s0"], quad_exp["s1"])
    quad_norm_add = plot(norm, quad_norm["s0"], quad_norm["s1"])

    lin_exp = linear_model(exp)
    lin_exp_stats = stats(lin_exp["s1"])
    print("\n Linear model with exp errors stats: ")
    stats_print(lin_exp_stats)

    lin_norm = linear_model(norm)
    lin_norm_stats = stats(lin_norm["s1"])
    print("\n Linear model with norm errors stats: ")
    stats_print(lin_norm_stats)

    lin_exp_add = plot(exp, lin_exp["s0"], lin_exp["s1"])
    lin_norm_add = plot(norm, lin_norm["s0"], lin_norm["s1"])


main()
