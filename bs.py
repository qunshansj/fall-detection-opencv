python

class Bs:
    def __init__(self):
        self.settings = settings.Settings()
        self.method = self.settings.bsMethod

        if self.method == 0:
            self.fgbg = cv2.BackgroundSubtractorMOG2(self.settings.MOG2history, self.settings.MOG2thresh, self.settings.MOG2shadow)
            self.foregroundMask = None

        if self.method == 1:
            self.backgroundFrame = None
            self.frameCount = 1

    def updateBackground(self, frame):  # Updating the background
        if self.method == 0:
            self.foregroundMask = self.fgbg.apply(frame, self.foregroundMask, self.settings.MOG2learningRate)

        if self.method == 1:
            alpha = (1.0/self.frameCount)
            if self.backgroundFrame is None:
                self.backgroundFrame = frame
            self.backgroundFrame = cv2.addWeighted(frame, alpha, self.backgroundFrame, 1.0-alpha, 0)
            self.frameCount += 1

    def compareBackground(self, frame): 
        if self.method == 0:
            return self.foregroundMask

        if self.method == 1:
            self.frameDelta = cv2.absdiff(self.backgroundFrame, frame)
            self.foregroundMask = cv2.threshold(self.frameDelta, self.settings.thresholdLimit, 255, cv2.THRESH_BINARY)[1]  # Creating a foregroundMask
            return self.foregroundMask

    def deleteBackground(self):  
        if self.method == 0:
            self.foregroundMask = None

        if self.method == 1:
            self.backgroundFrame = None

    def resetBackgroundIfNeeded(self, frame):
        if self.method == 0:
            if self.foregroundMask is None:
                self.foregroundMask = self.fgbg.apply(frame)

        if self.method == 1:
            if self.backgroundFrame is None:
                self.updateBackground(frame)
                self.frameCount = 1
