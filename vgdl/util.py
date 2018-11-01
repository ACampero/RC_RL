from IPython import embed
import itertools
import os
import random
import csv
import cPickle

ALNUM = '0123456789bcdefhijklmnpqrstuvwxyzQWERTYUIOPSDFHJKLZXCVBNM,./;[]<>?:`-=~!@#$%^&*()_+'
CHARS = 'bcdefhijklmnpqrstuvwxyzQWERTYUIOPSDFHJKLZXCVBNM'
CAPCHARS = 'QWERTYUIOPSDFHJKLZXCVBNM'

def softmax(w, t = 1.0):
	e = np.exp(np.array(w) / t)
	dist = e / np.sum(e)
	return dist

def normalize(array):
	z = float(sum(array))
	if z == 0:
		return [1./len(array)]*len(array) #if all items have the same score of 0, return the same score for all.
	else:
		return [a/z for a in array]

def manhattanDist(a, b):
	return abs(a[0]-b[0])+abs(a[1]-b[1])

def factorize(rle, n):
	## Decomposes into a list of numbers that are incides of [avatar, rle._obstypes.keys()]
	## that correspond to which indices are present in n
	## this follows the convention of rle._getSensors(), which won't report two of the same number as being in a location, so
	## the decomposition is unique (e.g., if n=4, this is because the decomposition is [4], rather than having to worry that it
	## would be [2,2]).
	orig_n = n
	decomposition = []
	if n%2==1:
		decomposition.append(0)
		n = n-1
	i = len(rle._obstypes.keys())
	while i>0:
		if n>=2**i:
			decomposition.append(i)
			n = n-2**i
		i = i-1

	return decomposition

def findNearestSprite(sprite, spriteList):
	## returns the sprite in spriteList whose location best matches the location of sprite.
	return sorted(spriteList, key=lambda x:abs(x.rect[0]-sprite.rect[0])+abs(x.rect[1]-sprite.rect[1]))[0]
		
def objectsToSymbol(rle, objects, symbolDict):
	objects = [rle._game.sprite_groups[o][0].colorName for o in objects]
	try:
		if len(objects)==1:
			if objects[0] not in symbolDict.keys():
				idx = len(symbolDict.keys())
				symbolDict[objects[0]] = ALNUM[idx]
			return symbolDict[objects[0]]
		else:
			for item in itertools.permutations(objects):
				if tuple(item) in symbolDict.keys():
					return symbolDict[tuple(item)]

		if not any([tuple(k) in symbolDict.keys() for k in list(itertools.permutations(objects))]):
			idx = len(symbolDict.keys())
			symbolDict[tuple(objects)] = ALNUM[idx]
			return ALNUM[idx]
	except:
		# import ipdb; ipdb.set_trace()
		print "objectsToSymbol problem."
		embed()

def getObjectColor(objectID, all_objects, game, colorDict):
	if objectID is None:
		return None
	elif objectID == 'EOS':
		return 'ENDOFSCREEN'
	elif objectID in all_objects.keys():
		return all_objects[objectID]['type']['color']
	elif objectID in game.getObjects().keys():
		return game.getObjects()[objectID]['type']['color']
	elif objectID in [colorDict[k] for k in colorDict.keys()]:
		# If we were passed a color to begin with (i.e., in the case of EOS)
		return objectID
	elif objectID in game.sprite_groups.keys():
		return colorDict[str(game.sprite_groups[objectID][0].color)]
	elif objectID in [obj.ID for obj in game.kill_list]:
		objectColor = [obj.color for obj in ame.kill_list
			if obj.ID==objectID][0]
		return colorDict[str(objectColor)]
	else:
		# for some reason we haven't been passed an ID but rather a sprite object
		objectName = objectID.name
		color = [all_objects[k]['type']['color'] for k in all_objects.keys() if all_objects[k]['sprite'].name==objectName][0]
		return color

def extendColorDict(num):
	for i in range(num):
		colorName = make_random_name(CAPCHARS)
		color = (random.choice(range(256)), random.choice(range(256)), random.choice(range(256)))
		print colorName + '=' + str(color)
		colorDict[str(color)] = colorName
	print colorDict

def make_random_name(chars):
	import random
	name = ''
	for i in range(6):
		name+=random.choice(chars)
	return name

def write_to_csv(foldername, filename, game):
	dirname = 'model_results'
	if dirname not in os.listdir('.'):
		os.makedirs(dirname)
	if filename not in os.listdir(dirname+'/'+foldername+'/'):
		f = open(dirname+'/'+foldername+'/'+filename, 'w+') #newfile and write
		writer = csv.writer(f)
		writer.writerow(('subject', 'condition', 'gameName', 'levels_won', 'steps', 'planner_steps', 'score'))
	else:
		f = open(dirname+foldername+'/'+filename, 'a+') ##append, but also read.
		writer = csv.writer(f)
	episodes = game['episodes']
	steps, levels_won, score, planner_steps = 0, 0, 0, 0
	for episode in episodes:
		steps += episode[1]
		planner_steps += episode[-1]
		levels_won += episode[2]
		if episode[3] is not None:
			score +=episode[3]
		else:
			score = None
		writer.writerow((game['modelType'], game['condition'], game['gameName'], levels_won, steps, planner_steps, score))
	f.close()
def ccopy(obj):
	return cPickle.loads(cPickle.dumps(obj))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
