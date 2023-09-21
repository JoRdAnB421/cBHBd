import time

import astropy.cosmology
import numpy as np
from p_tqdm import p_tqdm

from bhbdynamics import run_model

print(f"---------------- RUN BEGIN ----------------")

run_start_time = time.time()

# Redshift (using Planck 2018 cosmology)
z_step = 0.5
z_arr = np.arange(10, 0 - z_step, -z_step)
tf0_arr = astropy.cosmology.Planck18.lookback_time(z_arr).value

# Mass
M_step = 0.1
Mcl0_arr = 10 ** np.arange(np.log10(1e2), np.log10(2e7), M_step)  # Including endpoint

# Metallicity
Z_files_arr, Z_arr = [], []
with open("data/BHs/metallicity.txt", "r") as f:
    for line in f.readlines():
        Z_file, Z = line.split()
        Z_files_arr.append(Z_file)
        Z_arr.append(float(Z))

Mcl0_arr = np.flip(Mcl0_arr)

seeds = np.arange(10001, 10002, 1, dtype=int)

for seed in seeds:
    np.random.seed(seed)

    input_pars = []
    for Z_file, Z in zip(Z_files_arr, Z_arr):
        for tf0 in tf0_arr:
            for M in Mcl0_arr:
                input_pars.append([tf0, M, [Z_file, Z]])
    # res = map(run_model, input_pars)  # Use this for non-parallel runs
    res = p_tqdm.p_umap(run_model, input_pars)

    with open(f"runs/mergers_{seed}.txt", "w") as f:
        for full_bhout in res:
            for bhout_i in full_bhout:
                f.write("\t".join([str(el) for el in bhout_i]) + "\n")

run_end_time = time.time()
print(f"\n Run time: {run_end_time - run_start_time:.1f} s")

print(f"---------------- RUN END ----------------")
