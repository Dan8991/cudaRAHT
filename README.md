# cudaRAHT
cuda implementation of RAHT in python

# Installation
```
git clone git@github.com:Dan8991/cudaRAHT.git
cd cudaRAHT
mkdir dataset
conda env create -f environment.yaml
conda activate cudaRAHT
```

At this point you just need to move the point cloud long.ply used for testing in the dataset folder

# Running

If you want to run the full simulations used to obtain the plots run 

```
python run_simulation.py
```

with the desired number of iterations (the more the better since the average measurement will be more precise)

while if you only want to run the main you need to use the kernprof tool because the @profile 
decorator only works with kernprof. So you need to run 
```
kernprof -l main.py
```
and then append the extra arguments as follows
```
kernprof -l main.py --type=sequential --resolution=512 --stop_level=9
```
