import mediapipe as mp

class ImageLandmarker:
    def __init__(self):
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.createLandmarker()
      
    def createLandmarker(self):
        model_path = 'hand_landmarker.task'
        # Create a hand landmarker instance with the live stream mode:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE)
        
        self.landmarker = self.landmarker.create_from_options(options)
    
    def detect(self, mp_image):
        return self.landmarker.detect(image = mp_image)
        
    def close(self):
        self.landmarker.close()