import sys, ast, time
from theory_template import Game, TimeStep
from IPython import embed


def runInduction(vgdlString, gameOutput):
   
    verbose = True

    g = Game(vgdlString)
    trace = ([TimeStep(tr['agentAction'], tr['agentState'], tr['effectList'], tr['gameState']) for tr in gameOutput[0]],gameOutput[1])

    # start = time.time()
    hypotheses=list(g.induction(trace, verbose))
    # end = time.time()
    
    return hypotheses

def runInduction_DFS(vgdlString, gameOutput, maxTheories):

    verbose = True

    g = Game(vgdlString) # Use specific game specifications

    trace = ([TimeStep(tr['agentAction'], tr['agentState'], tr['effectList'], tr['gameState']) for tr in gameOutput[0]],gameOutput[1])
    

    ##TODO: currently (until Jackie fixes this), each timestep.events (which basically has as contents tr['effectList']) is being fed object IDs, 
    ## rather than object type). For now, changing this to get the object color so that theory induction can run as it was deisgned to run;
    ## Later change this to use some kind of an index for object type (right now 'type' lists the dimensions that we decided determine object type,
    ## but we need a string that is an ID that maps to each type.)
    def getObjectType(timestep, objectID, timesteps):
        objType = None
        for k in timestep.gameState['objects'].keys():
            if objectID in [o['ID'] for o in timestep.gameState['objects'][k].values()]:
                objType = k
                return objType
        if objType==None:
            for t in timesteps:
                for k in t.gameState['objects'].keys():
                    if objectID in [o['ID'] for o in t.gameState['objects'][k].values()]:
                        objType = k
                        return objType
    #trace[0] contains all timesteps
    for i in range(len(trace[0])):
        timestep = trace[0][i]
        for j in range(len(timestep.events)):
            event = timestep.events[j]
            print event
            if len(event)==3:
                timestep.events[j] = (event[0], getObjectType(timestep, event[1], trace[0]), getObjectType(timestep, event[2], trace[0]))
            elif len(event)==2:
                timestep.events[j] = (event[0], getObjectType(timestep, event[1], trace[0]))



    # start = time.time()

    hypotheses=list(g.runDFSInduction(trace, maxTheories, verbose))
    embed()

    # end = time.time()
    
    return g, hypotheses


if __name__ == "__main__":
    """
    Run: "python induction.py simpleGame1.txt simpleGame1_game_output.txt" 
    """
    game = sys.argv[1]
    output = sys.argv[2]
    with open("../vgdl_text/{}".format(game), 'r') as vf:
        vgdlString = ast.literal_eval(vf.read())

    with open("../output/{}".format(output), 'r') as f:
        output = f.readline()
        output_tuple = ast.literal_eval(output)
        #print output_tuple

    maxTheories = 100
    game, hypotheses = runInduction_DFS(vgdlString, output_tuple, maxTheories)


    embed()


