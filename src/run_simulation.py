import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

n_tests=1
modes = ["fullcuda", "numpy"]#, "partialsequential", "sequential"]
mapping = {
    "fullcuda": 0,
    "numpy": 1,
    "partialsequential": 3,
    "sequential": 3 
} 
times = {mode: [] for mode in modes}
for res_exp in tqdm(range(5, 11)):
    resolution = 2 ** res_exp
    for mode in modes:
        total_time = 0
        for i in range(n_tests):
            command = f"kernprof -l main.py --type={mode}"
            command += f" --resolution={resolution} --stop_level={res_exp - 1} >/dev/null 2>&1"
            os.system(command)
            with open("main.py.lprof", "rb") as f:
                data = pickle.load(f)

            time_info = data.timings[list(data.timings.keys())[mapping[mode]]]
            run_time = sum([x[2] for x in time_info])
            total_time += run_time
        times[mode].append(total_time/n_tests/10**6)

for mode in modes:
    plt.plot(range(5, 11), times[mode], label=mode)
plt.legend()
plt.show()


