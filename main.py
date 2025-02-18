import time

import astropy.cosmology
import numpy as np
from p_tqdm import p_tqdm
from tqdm import tqdm
import os
from bhbdynamics import run_model

# Make runs directory if doesn't exist
cwd = os.getcwd()
path = os.path.join(cwd, 'runs')

if not os.path.exists(path): os.mkdir(path)

print(f"---------------- RUN BEGIN ----------------")

run_start_time = time.time()

# Redshift (using Planck 2018 cosmology)
z_step = 0.5
z_arr = np.arange(10, 0 - z_step, -z_step)
tf0_arr = astropy.cosmology.Planck18.lookback_time(z_arr).value
# tf0=tf0_arr[0]
# Mass
M_step = 0.1
Mcl0_arr = 10 ** np.arange(np.log10(1e4), np.log10(2e7), M_step)  # Including endpoint

# Mcl0_arr = np.array([1e5])
# Mcl10_arr = np.array([1e4, 1e5, 1e6, 1e7])

# Metallicity
Z_files_arr, Z_arr = [], []
with open("data/BHs/3_metal.txt", "r") as f:
    for line in f.readlines():
        Z_file, Z = line.split()
        Z_files_arr.append(Z_file)
        Z_arr.append(float(Z))

Mcl0_arr = np.flip(Mcl0_arr)

# Header names for output file
head = ['mergertime', 'simtime', 'm1', 'm2', 'e', 'Mcl',
        'mergetype', 'tlbform', 'rh', 'vk', 'vesc', 'chi_f',
        's1', 's2', 'Z', 'a', 'genmerge', 'mbh']
writeHead = '\t'.join(map(str, head))+'\n'

seeds = np.arange(1, 2, 1, dtype=int)

# Save the input cluster properties for merger rate computation later
with open(f'Testing/ClusterInput_Seeds-{seeds[0]}-{seeds[-1]}.txt','w') as f:
    f.write(f'Cluster Mass : {list(Mcl0_arr)}\n')
    f.write(f'Metallicity : {Z_arr}\n')
    f.write(f'Formation Time : {list(tf0_arr)}\n')
    f.write(f'Redshift : {list(z_arr)}')

input()
for seed in seeds:
    np.random.seed(seed)

    input_pars = []

    for Z_file, Z in tqdm(zip(Z_files_arr, Z_arr), desc='metallicity'):
        for tf0 in tqdm(tf0_arr, desc='lookback time', leave=False):
            for M in tqdm(Mcl0_arr, desc='cluster mass', leave=False):
                input_pars.append([tf0, M, [Z_file, Z]])
    res = map(run_model, input_pars)  # Use this for non-parallel runs
    # res = p_tqdm.p_umap(run_model, input_pars)

    # Speed up the writing process by creating multiple lines at once and saving
    bufferSize=2000
    buffer=[]
    with open(f"Testing/mergers_{seed}.txt", "w") as f:
        f.write(writeHead)
        for full_bhout in tqdm(res):
            for bhout_i in full_bhout:
                buffer.append("\t".join(map(str, bhout_i)) + "\n")

                print(len(buffer), end='\r')
                # fush the buffer periodically
                if len(buffer)>=bufferSize:
                    f.writelines(buffer)
                    buffer.clear()

        # Write remaining buffer
        if buffer:
            f.writelines(buffer)
run_end_time = time.time()
print(f"\n Run time: {run_end_time - run_start_time:.1f} s")

print(f"---------------- RUN END ----------------")
