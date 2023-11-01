import time, threading

class FaceIdentity:
    
    def __init__(self, id=-1, name="", confidence=0.0, faceBox = None):
        self.name = name
        self.confidence = confidence
        self.id = id
        self.face = faceBox
        self.InRecognition = False
        self._recognizedNames = []
    
    def getName(self):
        return self.name
    
    def setName(self, name):
        self.name = name
        if(self.isInRecognition()):
            return self._recognizedNames.append(name)
        
    def getConfidence(self):
        return self.confidence

    def setConfidence(self, confidence):
        self.confidence = confidence
        
    def getID(self):
        return self.id
    
    def getFace(self):
        return self.face
    
    def setFace(self, face):
        self.face = face
    
    def startRecognition(self, callback, seconds):
        self.InRecognition = True
        self.timer = threading.Timer(seconds, callback)
        self.timer.start()
        
    def stopRecognition(self, action=None):
        self.InRecognition = False
        if self.timer:
            self.timer.cancel()
        if action is not None:
            action()
    
    def isInRecognition(self):
        return self.InRecognition
        
    def __str__(self):
        return f"FaceIdentity(id={self.id}, name={self.name}, confidence={self.confidence})"
    