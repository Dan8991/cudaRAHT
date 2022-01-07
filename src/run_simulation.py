import os
import pickle
from tqdm import tqdm

n_tests = 10
total_time = 0
for i in tqdm(range(n_tests)):
    os.system("kernprof -l main.py >/dev/null 2>&1")
    with open("main.py.lprof", "rb") as f:
        data = pickle.load(f)

    time_info = data.timings[list(data.timings.keys())[0]]
    run_time = sum([x[2] for x in time_info])
    total_time += run_time

print(f"total time: {total_time/n_tests / 10 ** 6}s")

