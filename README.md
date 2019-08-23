Interfacing with VGDL
=====================

Instructions to interface with VGDL and to run DeepRL Models.

Gym-like API to run your own models
--------------------
The Methods are defined in `VGDLEnvAndres.py` :
```
#Configs to set
self.trial_num = 1002
self.record_flag = 0 #1 to generate VGDL files (i.e. reward_histories used for figures)

#Attributes
self.game_name
self.game_over
self.action_space
self.observation_space

#Methods
self.step(action)
self.reset()
self.set_level(level,steps)
self.get_level()
```




To run Deep RL models
-------------------------
**Installation**

```
sudo apt-get update && sudo apt-get install python python2.7 python-pip virtualenv git wget emacs cmake zlib1g-devcmake zlib1g-dev

pip install wheel
pip install -r requirements

#FOR Dopamine:
pip install absl-py atari-py gin-config gym opencv-python tensorflow-gpu
```



**Run Dopamine**

Clone this Repo and our version of Dopamine from https://github.com/ACampero/dopamine as a submodule as in the Repo

```
python -um dopamine.discrete_domains.train \
  --base_dir=./tmp/dopamine/aliens \
  --gin_files='dopamine/agents/rainbow/configs/rainbow_aaaiAndres.gin' \
  --gin_bindings='create_atari_environment.game_name="VGDL_aliens"'
```

**Run Pytorch Implemenation of DDQN**

```
python run.py -game_name aliens
```








