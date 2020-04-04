from random import randint
import time

class MyPerson:
    tracks = []
    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.confidence = 0
    def getConfidence(self):
        return self.confidence
    def getRGB(self):
        return (self.R,self.G,self.B)
    def getTracks(self):
        return self.tracks
    def getId(self):
        return self.i
    def setState(self, newState):
        self.state = newState
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getAge(self):
        return self.age
    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x,self.y])
        self.x = xn
        self.y = yn
    def getDone(self):
        return self.done
    def setDone(self):
        self.done = True
    def updateConfidence(self, confNew):
        self.confidence = confNew
    def timedOut(self):
        return self.done
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True