import mediapipe as mp
import time

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