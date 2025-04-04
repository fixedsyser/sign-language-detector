import mediapipe as mp
from util import draw_landmarks_on_image
from matplotlib import pyplot as plt
import random
import glob

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
    
# plain image detector without ASL logic
def image_detector(image):
    try:
        detector = ImageLandmarker()
        detection_result = detector.detect(image)
    finally: 
        detector.close()
                    # method in util.py
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    plt.axis('off')
    plt.imshow(annotated_image)

# helper method to get random image from ASL_letters
def get_random_image():
    filename = random.choice(glob.glob('./american-sign-language-letters.v1i.coco/train/*.jpg'))
    return mp.Image.create_from_file(filename)