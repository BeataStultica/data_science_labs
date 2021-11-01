import numpy as np
import math as mt
import matplotlib.pyplot as plt


def stats(data):
    dS = np.var(data)
    return {"median": np.median(data), "ds": dS, "scvS": mt.sqrt(dS)}


def stats_print(stats):
    for name, value in stats.items():
        print(f"{name}: {value}")


def normal(n=10000, dm=0, dsig=200):
    return ((np.random.randn(n))*dsig)+dm


def quad_model(error, n=10000, nAV=1000, dm=0, dsig=200):
    SAV = np.zeros((nAV))
    SSAV = np.zeros((nAV))
    SV0 = np.zeros((n))
    SV_AV = np.zeros((n))
    S = np.zeros((n))
    for i in range(n):
        S[i] = np.random.randint(0, n)
    mS = np.median(S)
    dS = np.var(S)
    scvS = mt.sqrt(dS)
    for i in range(nAV):
        # рівномірний розкид номерів АВ в межах вибірки розміром 11-n (штучно створюємо 10 перших вимірів без аномалій)
        SAV[i] = mt.ceil(np.random.randint(11, n))
    S0 = np.zeros((n))
    S1 = np.zeros((n))
    for i in range(n):
        S0[i] = (10+0.00009*i*i)
        S1[i] = S0[i]+error[i]
        SV0[i] = abs(S1[i] - S0[i])
        SV_AV[i] = S1[i]
    # аномальна випадкова похибка з нормальним законом
    SSAV = np.random.normal(dm, (3*dsig), nAV)
    for i in range(nAV):
        k = int(SAV[i])
        SV_AV[k] = S0[k] + SSAV[i]
    return {"s0": S0, "s1": S1, "sv0": SV0, "sv_av": SV_AV}


# добавив вибір степеня полінома
def MNK(S, iter=10000, step=2):
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, step+1))
    for i in range(iter):
        Yin[i, 0] = float(S[i])
        for j in range(1, step+1):
            F[i, j] = float(i**j)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    return {"yout": Yout, "yin": Yin}


def anomal_detect(S, window_start=0, window_end=10):
    #prev_scvS = float("inf")
    accum_anomal = []
    for i in range(window_end-window_start, len(S)):
        window = MNK(S[i-window_end:i], window_end-window_start)
        current_stats = stats(window["yout"])
        # if current_stats["scvS"] > abs(3*prev_scvS):
        if abs(S[i]-current_stats["median"]) > 3*current_stats["scvS"]:
            accum_anomal.append(i)
        #prev_scvS = current_stats["scvS"]

    return restore_anomal(S, accum_anomal)


def restore_anomal(S, anomals):
    S0 = np.zeros((len(S)))
    for i in range(len(S)):
        if i-1 not in anomals and i+1 not in anomals and i in anomals and i != len(S)-1 and i != 0:
            S0[i] = (S[i-1]+S[i+1])/2
        elif i not in anomals:
            S0[i] = S[i]
        else:
            window = MNK(S0[i-10:i], 10, step=1)["yout"]
            S0[i] = window[-1] + abs(window[0]-window[-1])/10
    return S0


def plot(error, S0, S1, SV_AV, yout, yin):
    # закон розподілу випадкової помилки
    print("\nСтатистичні характеристики законy розподілу випадкової помилки")
    error_st = stats(error)
    stats_print(error_st)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.hist(error, bins=20, facecolor="blue", alpha=0.5)
    ax1.set_title("Випадкова помилка")

    # вхідна зашумлена вибірка без аномалій
    print("\nСтатистичні характеристики вхідної зашумленої вибірки без аномалій")
    S1_st = stats(S1)
    stats_print(S1_st)

    ax2.hist(S1, bins=20, alpha=0.5, label='S1')
    ax2.set_title("Зашумлена вхідна вибірка без аномалій")

    # аномальна вибірка
    print("\nСтатистичні характеристики вхідної зашумленої вибірки з аномаліями")
    SV_st = stats(SV_AV)
    stats_print(SV_st)
    ax3.hist(SV_AV)
    ax3.set_title("Вибірка з аномаліями")

    # згладжена вибірка
    print("\nСтатистичні характеристики згладженої вибірки")
    yout_st = stats(yout)
    stats_print(yout_st)
    ax4.hist(yout)
    ax4.set_title("Згладжена вибірка")
    plt.show()

    # графіки
    fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2)
    ax5.plot(S1)
    ax5.plot(S0)
    ax5.set_title("Квадратичний тренд та зашумлена без аномалій вибірка")
    ax6.plot(SV_AV)
    ax6.set_title("Вибірка з аномаліями")

    ax7.plot(yin)
    ax7.plot(yout)
    ax7.set_title("Зглажена вибірка після відновлення аномалій")
    plt.show()


def main():
    norm = normal()
    quad_norm = quad_model(norm)
    S_no_anomals = anomal_detect(quad_norm["sv_av"])
    mnk = MNK(S_no_anomals)
    quad_norm_add = plot(
        norm, quad_norm["s0"], quad_norm["s1"], quad_norm["sv_av"], mnk["yout"], mnk["yin"])


main()
