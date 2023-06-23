from random import randint

import numpy as np

INPUT_FILE = "./data/ex_input.txt"
IMMUTABLE_COLS = 3
ATTACK_ITER = 15

# load initial input -- this data is assumed valid
original = np.loadtxt(INPUT_FILE, delimiter=',')
num_rows, num_cols = original.shape

# (randomly for this example) choose some columns to mask:
indices_immutable = sorted(list(set(
    [randint(0, num_cols - 1) for _ in range(IMMUTABLE_COLS)])))

# Create mask vector where 0 = immutable and 1 = mutable
mask_vector = np.array([0 if i in indices_immutable else 1
                        for i in range(0, num_cols)])
mask = np.tile(mask_vector, (num_rows, 1))

# initialize adversarial, same shape and values as input
adv = np.copy(original)

# simulate "adversarial attack" perturbation
# (this will just add random noise)
for _ in range(ATTACK_ITER):
    noise = np.random.rand(*adv.shape)

    # enforce immutable
    adv = (adv + noise) * mask + original * (1 - mask)

print("Indices", indices_immutable, 'are immutable:')
# Take the diff, coerce to int type for display

for row in (adv - original).astype(int):
    print(' '.join(['=' if e == 0 else '.' for e in row]))
