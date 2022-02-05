import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

n_tests=1
modes = ["fullcuda", "numpy", "sequential"]
mapping = {
    "fullcuda": 0,
    "numpy": 1,
    "sequential": 3 
} 
times = {mode: [] for mode in modes}
#iterating through the resolutions
for res_exp in tqdm(range(5, 11)):
    resolution = 2 ** res_exp
    for mode in modes:
        total_time = 0
        for i in range(n_tests):
            #using kernprof to profile the code
            command = f"kernprof -l main.py --type={mode}"
            command += f" --resolution={resolution} --stop_level={res_exp - 1} >/dev/null 2>&1"
            os.system(command)
            with open("main.py.lprof", "rb") as f:
                data = pickle.load(f)

            #getting the time outputted by kernprof
            time_info = data.timings[list(data.timings.keys())[mapping[mode]]]
            run_time = sum([x[2] for x in time_info])
            total_time += run_time
        times[mode].append(total_time/n_tests/10**6)

for mode in modes:
    plt.plot(range(5, 5 + len(times[mode])), times[mode], label=mode)
plt.ylabel("time (s)")
plt.xlabel("resolution exponent")
plt.legend()
plt.show()

# from the 1024 resolution trying to find out how much the stop level influences the result
times["fullcuda"] = []
for stop_level in tqdm(range(1, 9)):
    total_time = 0
    for i in range(n_tests):
        command = f"kernprof -l main.py --type=fullcuda"
        command += f" --resolution={1024} --stop_level={stop_level} >/dev/null 2>&1"
        os.system(command)
        with open("main.py.lprof", "rb") as f:
            data = pickle.load(f)

        time_info = data.timings[list(data.timings.keys())[mapping["fullcuda"]]]
        run_time = sum([x[2] for x in time_info])
        total_time += run_time
    times["fullcuda"].append(total_time/n_tests/10**6)

plt.plot(range(3, 27, 3), times["fullcuda"])
plt.ylabel("time (s)")
plt.xlabel("# of collapse operations")
plt.show()

