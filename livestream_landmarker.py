import mediapipe as mp
import cv2 as cv
import time

from util import draw_landmarks_on_image

class LivestreamLandmarker:
    def __init__(self):
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()
      
    def createLandmarker(self):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            # print('hand landmarker result: {}'.format(result))
            self.result = result

        model_path = 'hand_landmarker.task'
        # Create a hand landmarker instance with the live stream mode:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=update_result)
        
        self.landmarker = self.landmarker.create_from_options(options)
    
    def detect_async(self, mp_image):
        frame_timestamp_ms = time.time_ns() // 1_000_000
        self.landmarker.detect_async(image = mp_image, timestamp_ms = frame_timestamp_ms)
        
    def close(self):
        self.landmarker.close()

# a plain landmark detector without ASL logic        
def livestream_landmark_detector():
    try:
        cap = cv.VideoCapture(0)
        detector = LivestreamLandmarker()
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detector.detect_async(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detector.result)
            cv.imshow('frame', annotated_image)
            # 1000/50 = 20 FPS
            if cv.waitKey(100) == ord('q'):
                break
    finally:    
        # When everything done, release the capture
        detector.close()
        cap.release()
        cv.destroyAllWindows()