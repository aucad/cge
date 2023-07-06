import numpy as np


def show(m, label=None):
    print(label if label else '', end='\n' if label else '')
    print('\n'.join([' '.join([str(e) + ' ' for e in r]) for r in m]) + '\n')


# create arrays:
original = np.array(
    [[4, 1, 0, 2, -7, 6], [5, 3, 3, 8, 4, 0], [1, 8, 7, 6, 5, 6],
     [4, 5, 4, 1, -4, 3], [2, 2, 3, 3, 8, 1]])

# original + random noise
adversary = np.copy(original) + np.random.randint(2, size=original.shape)

# mask is initially all 1's
mask = np.ones(original.shape, dtype=np.ubyte)

# some values get set to 0
mask[:, 4] = 0  # entire feature
mask[1, 5] = 0  # select records
mask[1:4, 2] = 0  # select records
mask[0:3, 3] = 0  # select records


# enforce constraints:
adv_prime = adversary * mask + original * (1 - mask)

show(original, 'Original')
show(adversary, 'Adversary')
show(mask, 'mask')
show(adv_prime, 'returned adversary')

# lambda x: x == 0 or x == 1
# lambda x: x * (2 ** 64) % 12 == 0
