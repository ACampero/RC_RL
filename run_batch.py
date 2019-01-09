import subprocess

games = ['aliens_variant_1']

for game_name in games:
	subprocess.call("sbatch rl_runscript.sh {}".format(game_name), shell = True)