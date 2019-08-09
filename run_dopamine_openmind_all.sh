#!/bin/bash
#SBATCH --qos=tenenbaum

for i in {1..90}
do
	sbatch ./run_dopamine_openmind.sh $i
done
