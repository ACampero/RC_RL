Openmind Instructions
=====================

Instructions to train this repo's models on Openmind inside containers.

Step 1. Folder setup
--------------------

For these scripts to work properly, setup is as follows:
```
mkdir /om2/user/$USER/vgdl_testing
cd /om2/user/$USER/vgdl_testing
git clone https://github.com/lcary/RC_RL
cd RC_RL
```

For other filesystem setups, modify the path in the 2 scripts in this folder accordingly.

Step 2. Get the container
-------------------------

Download the container image titled `VGDLContainer.py2.simg` from:
https://github.mit.edu/lcary/singularity-builds/releases/tag/dopamine

Either download directly to openmind (e.g. via `wget`)
or download locally and `scp` or `rsync` to Openmind.

Copy the container image into the clone of this repo from the previous step.

Step 3. Install Python dependencies
-----------------------------------

Install the python dependencies by running these commands:
```
cd RC_RL
module add openmind/singularity
singularity shell --bind $(pwd) VGDLContainer.py2.simg
virtualenv venv
source venv/bin/activate
pip install wheel
pip install -r requirements.txt
```

This is a one-time setup step.

Step 4. Train the model
-----------------------

The following command submits a job to train a model.
```
sbatch ./openmind/run_model_sbatch.sh
```

Output will be saved to the `output/` folder, errors to `errors/`.
