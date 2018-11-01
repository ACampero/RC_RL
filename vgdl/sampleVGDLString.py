
push_game = """
BasicGame frame_rate=30
    SpriteSet        
        hole   > ResourcePack color=LIGHTBLUE
        avatar > MovingAvatar color=DARKBLUE #cooldown=4 
        box    > ResourcePack 
            box1 > color=ORANGE               
        treasure > ResourcePack color=GREEN limit=5
        goal > Passive color=GOLD
        trap > ResourcePack color=RED limit=5
        cloud > Passive color=BLUE
        medicine > Resource limit=3 color=WHITE
        poison > Resource limit=3 color=BROWN
        wall > Immovable color=BLACK      
        score > Resource color=PINK limit=10         
    LevelMapping
        0 > hole
        1 > box1
        3 > treasure 
        t > trap    
        c > cloud 
        m > medicine
        p > poison
        w > wall   
        g > goal 
        h > hole
    InteractionSet
        avatar wall > stepBack  
        hole avatar > killSprite
        treasure avatar > changeResource resource=score value=5
        treasure avatar > killSprite
        trap avatar > changeResource resource=score value=-5
        trap avatar > killSprite
        cloud avatar > killSprite
        avatar medicine > changeResource resource=medicine value=1
        medicine avatar > killSprite
        avatar poison > changeResource resource=medicine value=-1
        poison avatar > killSprite
        avatar poison > killIfHasLess resource=medicine limit=-1
        box avatar  > bounceForward
        box wall    > undoAll        
        box box     > undoAll
        box hole    > killSprite
        box treasure > undoAll
        box poison > undoAll
        box medicine > undoAll
        goal avatar > killSprite  
    TerminationSet
        SpriteCounter stype=box     limit=0 win=True
        SpriteCounter stype=goal    limit=0 win=True
        SpriteCounter stype=avatar  limit=0 win=False          
"""

mario_game = """
BasicGame
    SpriteSet 
        elevator > Missile orientation=UP speed=0.1 color=BLUE
        moving > physicstype=GravityPhysics
            avatar > MarioAvatar airsteering=True
            evil   >  orientation=LEFT
                goomba     > Walker     color=BROWN 
                paratroopa > WalkJumper color=RED
        goal > Immovable color=GREEN
            
    TerminationSet
        SpriteCounter stype=goal      win=True     
        SpriteCounter stype=avatar    win=False     
           
    InteractionSet
        evil avatar > killIfFromAbove scoreChange=1
        avatar evil > killIfAlive
        moving EOS  > killSprite 
        goal avatar > killSprite
        moving wall > wallStop friction=0.1
        moving elevator > pullWithIt        
        elevator EOS    > wrapAround
        
    LevelMapping
        G > goal
        1 > goomba
        2 > paratroopa
        = > elevator
"""
