python

class MotionDetection:
    def __init__(self):
        self.cap = None
        self.fgbg = cv2.createBackgroundSubtractorKNN()
        self.consecutive_frames = 0

    def select_file(self):
        file_path = filedialog.askopenfilename()
        return file_path

    def use_webcam(self):
        self.cap = cv2.VideoCapture(0)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        if contours:
            areas = []
            heights = []
            widths = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                ar = cv2.contourArea(contour)
                areas.append(ar)
                heights.append(h)
                widths.append(w)

            max_area = max(areas or [0])
            max_area_index = areas.index(max_area)

            cnt = contours[max_area_index]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(fgmask, [cnt], 0, (255, 255, 255), 3)

            if h < w:
                self.consecutive_frames += 1

            aspect_ratio = h / w
            if self.consecutive_frames > 10 and aspect_ratio < 0.5:
                cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            if aspect_ratio >= 0.5:
                self.consecutive_frames = 0
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('video', frame)

            if cv2.waitKey(2) & 0xff == ord('q'):
                return False
        else:
            return False

        return True

    def run(self):
        root = tk.Tk()
        root.withdraw()

        mode = input("Enter 'webcam' to use webcam or 'file' to select a file: ")

        if mode == 'webcam':
            self.use_webcam()
        elif mode == 'file':
            file_path = self.select_file()
            self.cap = cv2.VideoCapture(file_path)
        else:
            print("Invalid mode. Exiting program.")
            exit()

        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            if not self.process_frame(frame):
                break

        self.cap.release()
        cv2.destroyAllWindows()

