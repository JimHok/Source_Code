from scipy.spatial.distance import jaccard, rogerstanimoto, canberra
import numpy as np


def shiftbits_ham(template, noshifts):
    templatenew = np.zeros(template.shape)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    if noshifts == 0:
        templatenew = template

    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew


def HammingDistance(template1, mask1, template2, mask2):
    hd = np.nan

    # Shifting template left and right, use the lowest Hamming distance
    for shifts in range(-50, 50):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits = template1s.size - nummaskbits

        C = np.logical_xor(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        bitsdiff = np.sum(C == 1)

        if totalbits == 0:
            hd = np.nan
        else:
            hd1 = bitsdiff / totalbits
            if hd1 < hd or np.isnan(hd):
                hd = hd1

    return hd


def JaccardDistance(template1, mask1, template2, mask2):

    bitsdiff_arr = []
    for _, shifts in enumerate(range(-50, 50)):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        template1n = np.logical_and(template1s, np.logical_not(mask))
        template2n = np.logical_and(template2, np.logical_not(mask))

        if np.isnan(template1n).all() or np.isnan(template2n).all():
            return np.nan

        bitsdiff_arr.append(
            jaccard(template1n.flatten(), template2n.flatten()))

    return min(bitsdiff_arr)


def TanimotoDistance(template1, mask1, template2, mask2):

    rog_dist = []
    for i, shifts in enumerate(range(-50, 50)):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        template1n = np.logical_and(template1s, np.logical_not(mask))
        template2n = np.logical_and(template2, np.logical_not(mask))

        rog_dist.append(rogerstanimoto(
            template1n.flatten(), template2n.flatten()))

    return min(rog_dist)


def WeightedEuclideanDistance(template1, mask1, template2, mask2):
    wed = np.nan

    std = []
    for i in range(len(template2)):
        tstd = []
        for j in range(len(template2[i])):
            tstd.append(np.std(template2[i]))
        std.append(tstd)
    # print(std)
    # print(len(std), " ", len(std[0]))
    bitsdiff_arr = np.empty(40, dtype=np.float64)
    totalbits_arr = np.empty(40, dtype=np.float64)

    for i, shifts in enumerate(range(-20, 20)):
        template1s = shiftbits_ham(template1, shifts)
        mask1s = shiftbits_ham(mask1, shifts)

        mask = np.logical_and(mask1s, mask2)
        nummaskbits = np.sum(mask == 1)
        totalbits_arr[i] = template1s.size - nummaskbits

        C = np.subtract(template1s, template2)
        C = np.logical_and(C, np.logical_not(mask))
        C = np.power(C, 2)
        C = np.multiply(C, std)
        bitsdiff_arr[i] = np.sum(C)

    for i, totalbits in enumerate(totalbits_arr):
        if totalbits == 0:
            wed = np.nan
        else:
            wed1 = (bitsdiff_arr[i]/(totalbits))
            if wed1 < wed or np.isnan(wed):
                wed = wed1

    return wed


def Canberra(template1, mask1, template2, mask2):
    template1n = []
    template2n = []
    template1 = template1.tolist()
    template2 = template2.tolist()
    allbit = []
    for i in range(64):
        for j in range(800):
            if (not mask2[i][j] and not mask1[i][j]):
                template1n.append(template1[i][j])
                template2n.append(template2[i][j])

    for i, shifts in enumerate(range(-8, 9)):
        C = canberra(template1n, template2n[i:]+template2n[:i])
        allbit.append(C)

    # print(len(template1n))

    return (min(allbit))
