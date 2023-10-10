#!/bin/bash

#SBATCH -p compute                  # Specify the partition or machine type used [compute/memory/gpu]
#SBATCH -N 1 --ntasks-per-node=32   # Specify the number of nodes and the number of core per node
#SBATCH --mem=64GB                  # Specify memory to use in the node or specify
#SBATCH -t 5-00:00:00                 # Specifies the maximum time limit (hour: minute: second)
#SBATCH -J Fusion-full               # Specify the name of the Job
#SBATCH -A pre0014                 # Specify Project account which will be received after Register

## purge and load modules
module purge                        # unload all modules as they may have previously been loaded.
module load Python/3.8.6-GCCcore-10.2.0                   # Load the module that you want to use. This example is intel

## your script
python Fusion.py                 # Run your program or executable code