import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
from .calculatePersistence import *
from .symplicalClasses import *
# import cv2
import matplotlib.pyplot as plt
import gudhi.wasserstein
import numpy as np
# from scipy.stats import wasserstein_distance

def q_distance(phi_1, phi_2, col = 100):
    phi_1 = phi_1[:, 0:col]
    phi_2 = phi_2[:, 0:col]

    # print(phi_1.shape)
    # print(phi_2.shape)

    r, c = phi_1.shape
    dis = 0

    for p1, p2 in zip(phi_1, phi_2):
        p1 = np.array([p1])
        p2 = np.array([p2])
        # print(p1.shape)
        # print(p2.shape)

        cleanSimple = calcPers(p1)
        cleanSimple = np.array(cleanSimple)
        noisySimple = calcPers(p2)
        noisySimple = np.array(noisySimple)
        dis += gudhi.wasserstein.wasserstein_distance(cleanSimple, noisySimple, order=1., internal_p=2.)


    # clean = np.array([[3.4, 3.9], [7.5, 7.8]])
    # noisy = np.array([[4.5, 1.4]])
    # print(type(clean))
    # print(len(clean))
    # print(type(noisy))
    # print(cleanSimple.shape)
    # print(noisySimple.shape)

    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(cleanSimple[:, 0], cleanSimple[:, 1], label='supp($p(x)$)')
    # plt.scatter(noisySimple[:, 0], noisySimple[:, 1], label='supp($q(x)$)')
    # plt.legend()
    # plt.xlim(0, cleanSimple.max()*1.2)
    # plt.ylim(0, cleanSimple.max()*1.2)
    # plt.show()

    # x = wasserstein_distance(cleanSimple[0], noisySimple[0], u_weights=None, v_weights=None)
    # print(x)
    return dis/r

def main():
    phi_1 = np.random.rand(100, 1)
    phi_2 = np.random.rand(100, 1)
    print("Wasserstein distance value = " + '%.4f' % q_distance(phi_1, phi_2))


if __name__ == '__main__':
    main()





