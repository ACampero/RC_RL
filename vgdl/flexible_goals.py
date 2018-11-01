

## Example changes you can make to play 'flexible goals' games. These changes correspond to the game expt_flexible goals.

## 1. avoid all blue things.

## find c4 c5 noveltyRule, delete it.
del self.hypotheses[0].terminationSet[11]
## Take away hypothesized c4 c5 rules
del self.hypotheses[0].interactionSet[17]
del self.hypotheses[0].interactionSet[14] #both c4 c5 rules


from vgdl.theory_template import InteractionRule
newRule = InteractionRule('killSprite', 'avatar', 'c3', {}, set(), generic=False)     
self.hypotheses[0].interactionSet.append(newRule)



## 2. get all blue things
del self.hypotheses[0].interactionSet[-1] ## killSprite avatar c3

del self.hypotheses[0].terminationSet[-1] ## whichever has c2 in it
											## and whichever has avatar=0 True in it.
from vgdl.theory_template import SpriteCounterRule
newRule = SpriteCounterRule("c3", 0, True) 
self.hypotheses[0].terminationSet.append(newRule)


## die as quickly as possible

del self.hypotheses[0].terminationSet[-1]   
del self.hypotheses[0].terminationSet[-2]   

## remove c5 c7 terminationRule

from vgdl.theory_template import SpriteCounterRule
newRule = SpriteCounterRule("avatar", 0, True)  
self.hypotheses[0].terminationSet.append(newRule)

del self.hypotheses[0].terminationSet[-2]
del self.hypotheses[0].terminationSet[-2]


from vgdl.theory_template import InteractionRule
newRule = InteractionRule('stepBack', 'c4', 'c5', {}, set(), generic=False)     ##add yellow darkgrey stepBack
self.hypotheses[0].interactionSet.append(newRule) 


