import pygame
import sys
sys.path.insert(0, '../')
from tools import Node, indentTreeParser
from collections import defaultdict
import os
import uuid
import subprocess
import glob
# import ipdb
from IPython import embed
from core import *
from tools import roundedPoints
from ontology import colorDict
# from vgdl.core import *
# from vgdl.tools import roundedPoints
# from vgdl.ontology import colorDict
from copy import deepcopy

class Sprite(object):
    """
    TODO: Incorporate properties into theory induction loop.
    """
    def __init__(self, vgdlType, color, className=None, args=None):
        self.vgdlType = vgdlType
        self.color = color
        self.className = className
        self.args = args

    # TODO: Should enforce proper syntax for properties
    def display(self):
        print (self.vgdlType, self.color, self.className, self.args)

    def __eq__(self, other):
        return all([
            self.vgdlType==other.vgdlType,
            self.color==other.color,
            self.className==other.className,
            self.args==other.args
            ])

    def __ne__(self, other):
        return not self.__eq__(other)

class SpriteParser(object):
    resourcePackTypeStrings = {'Immovable', 'Passive', 'ResourcePack', 'Spreader', 'Portal', 'SpawnPoint', 'Conveyor'}
    def __init__(self):
    	self.sprite_types = dict()

    def _eval(self, estr):
        """ Whatever is visible in the global namespace (after importing the ontologies)
        can be used in the VGDL, and is evaluated.
        """
        from ontology import * #@UnusedWildImport
        return eval(estr)

    def _parseArgs(self, s,  sclass=None, args=None):
        if not args:
            args = {}
        sparts = [x.strip() for x in s.split(" ") if len(x) > 0]
        if len(sparts) == 0:
            return sclass, args
        if not '=' in sparts[0]:
            sclass = self._eval(sparts[0])
            sparts = sparts[1:]
        for sp in sparts:
            k, val = sp.split("=")
            try:
                args[k] = self._eval(val)
            except:
                args[k] = val
        return sclass, args

    def parseGame(self, tree):
        """ Accepts either a string, or a tree. """
        if not isinstance(tree, Node):
            tree = indentTreeParser(tree).children[0]
        sclass, args = self._parseArgs(tree.content)
        self.game = sclass(**args)
        for c in tree.children:
            if c.content == "SpriteSet":
                self.parseSprites(c.children)
        #Return list of sprite types.
        return self.sprite_types.values()


    def parseSprites(self, snodes, parentclass=None, parentargs={}, parenttypes=[]):
        resourcePackTypes = {self._eval(obj_type) for obj_type in SpriteParser.resourcePackTypeStrings}
        resourceType = self._eval("Resource")
        EOS = "EOS"
        self.sprite_types[EOS] = Sprite(EOS,None,{})

        for sn in snodes:
            assert ">" in sn.content
            key, sdef = [x.strip() for x in sn.content.split(">")]
            sclass, args = self._parseArgs(sdef, parentclass, parentargs.copy())

            stypes = parenttypes+[key]
            if 'singleton' in args:
                if args['singleton']==True:
                    self.game.singletons.append(key)
                args = args.copy()
                del args['singleton']

            if len(sn.children) == 0:
                # print (sclass, args, stypes)
                if 'color' in args:
                    color_type = colorDict[str(args['color'])]
                    args_without_color = deepcopy(args)
                    del args_without_color['color']

                    #print "CLASS TYPE:", sclass

                    if sclass in resourcePackTypes:
                        #print "--> will be converted to ResourcePack"
                        self.sprite_types[key] = Sprite(self._eval('ResourcePack'), color_type, args_without_color)
                        # self.sprite_types[key] = (self._eval('ResourcePack'), colorized_args, stypes)
                    elif sclass == resourceType:
                        #print "--> will be converted to ResourcePack"
                        self.sprite_types[key] = Sprite(self._eval('ResourcePack'), color_type, args_without_color)
                        self.sprite_types[key+"_resource"] = Sprite(self._eval('ResourcePack'), color_type+"_resource", args_without_color)
                    else:
                        #print "--> will be ITSELF"
                        self.sprite_types[key] = Sprite(sclass, color_type, args_without_color)


                        # print self.sprite_types[key].vgdlType
                        # print self.sprite_types[key].color
                else:
                    args_without_color = deepcopy(args)
                    isResourceType = False
                    if sclass in resourcePackTypes:
                        s = self._eval('ResourcePack')
                        # self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args)
                    elif sclass == resourceType:
                        isResourceType = True
                        s = self._eval('ResourcePack')
                        # self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                        # self.sprite_types[key+"_resource"] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                    else:
                        s = sclass
                        self.sprite_types[key] = Sprite(sclass, None, args)

                    try:
                        # if s.color == None:
                        #     embed()

                        color = str(s.color)
                        if color in colorDict:
                            color = colorDict[color]

                        if isResourceType:
                            self.sprite_types[key] = Sprite(s, color, args_without_color)
                            self.sprite_types[key+"_resource"] = Sprite(s, color, args_without_color)
                        else:
                            self.sprite_types[key] = Sprite(s, color, args_without_color)

                    except AttributeError:
                        if isResourceType:
                            self.sprite_types[key] = Sprite(s, None, args_without_color)
                            self.sprite_types[key+"_resource"] = Sprite(s, None, args_without_color)
                        else:
                            self.sprite_types[key] = Sprite(s, None, args_without_color)

                        # if sclass in resourcePackTypes:
                        #     self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args)
                        #     # self.sprite_types[key] = (self._eval('ResourcePack'), colorized_args, stypes)
                        # elif sclass == resourceType:
                        #     self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                        #     self.sprite_types[key+"_resource"] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                        # else:
                        #     self.sprite_types[key] = Sprite(sclass, None, args)

                        # if sclass in resourcePackTypes:
                        #     self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args)
                        #     # self.sprite_types[key] = (self._eval('ResourcePack'), colorized_args, stypes)
                        # elif sclass == resourceType:
                        #     self.sprite_types[key] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                        #     self.sprite_types[key+"_resource"] = Sprite(self._eval('ResourcePack'), None, args_without_color)
                        # else:
                        #     self.sprite_types[key] = Sprite(sclass, None, args)

                if key in self.game.sprite_order:
                    # last one counts
                    self.game.sprite_order.remove(key)
                self.game.sprite_order.append(key)
            else:
                self.parseSprites(sn.children, sclass, args, stypes)
