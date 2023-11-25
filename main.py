python


class FallDetector:
    def __init__(self, video_path):
        self.video = video.Video(video_path)
        time.sleep(0.5)
        self.video.nextFrame()
        self.video.testBackgroundFrame()
    
    def run(self):
        while True:
            try:
                self.video.nextFrame()
                self.video.testBackgroundFrame()
                self.video.updateBackground()
                self.video.compare()
                self.video.showFrame()
                self.video.testSettings()
                if self.video.testDestroy():
                    sys.exit()
            except Exception as e:
                break

