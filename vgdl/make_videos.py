import cPickle
import os, subprocess, shutil
from IPython import embed
from core import VGDLParser

def makeMovies(directory):
	for pickleFile in [file for file in os.listdir(directory) if 'DS_Store' not in file]:
		with open(directory+'/'+pickleFile, 'r') as f:
			gameData = cPickle.load(f)
			gameName, gameString, levelString, episodes, param_ID = gameData['gameInfo']['gameName'], \
					gameData['gameInfo']['gameString'], gameData['gameInfo']['levelString'], gameData['episodes'], gameData['modelParams']
			for statesEncountered in episodes:
				makeImages(gameName, gameString, levelString, statesEncountered, param_ID)
			makeMovie(param_ID, gameName, pickleFile)
	return

def makeImages(gameName, gameString, levelString, statesEncountered, param_ID):
    VGDLParser.playGame(gameString, levelString, statesEncountered, \
        persist_movie=True, make_images=True, make_movie=False, movie_dir="videos/"+gameName, gameName = gameName, parameter_string=param_ID, padding=10)

def makeMovie(param_ID, gameName, filename):
    print "Creating Movie"
    movie_dir = "videos/{}/{}".format(param_ID, gameName)

    if not os.path.exists(movie_dir):
        print movie_dir, "didn't exist. making new dir"
        os.makedirs(movie_dir)
    # round_index = len([d for d in os.listdir(movie_dir) if d != '.DS_Store'])
    # video_dirname = movie_dir+"/round"+str(round_index)+".mp4"
    video_dirname = movie_dir+"/"+filename+".mp4"

    images_dir = "images/tmp/{}/%09d.png".format(gameName)
    com = "ffmpeg -i " +images_dir+ " -pix_fmt yuv420p -filter:v 'setpts=4.0*PTS' "+ video_dirname
    command = "{}".format(com)
    subprocess.call(command, shell=True)
    # empty image directory
    shutil.rmtree("images/tmp/"+gameName)
    os.makedirs("images/tmp/"+gameName)
    return

# embed()
## ## To convert all raw-video-data for 'scoretest' game:
dirname = 'raw_video_info/params__IW=2__ea=True/scoretest'
makeMovies(dirname)



# embed()
