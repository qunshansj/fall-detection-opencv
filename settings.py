python
class Settings(object):
    def __init__(self):
        self.debug = 1
        self.source = 0
        self.bsMethod = 1
        self.MOG2learningRate = 0.001
        self.MOG2shadow = 0
        self.MOG2history = 100
        self.MOG2thresh = 20
        self.minArea = 50*50
        self.thresholdLimit = 20
        self.dilationPixels = 30
        self.useGaussian = 1
        self.useBw = 1
        self.useResize = 1
        self.gaussianPixels = 31
        self.movementMaximum = 75
        self.movementMinimum = 3
        self.movementTime = 35
        self.location = 'Viikintie 1'
        self.phone = '01010101010'
