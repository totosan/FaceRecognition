import cv2
import numpy as np
import os
from skimage import transform
from PIL import Image
from face_check import FaceCheck

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
  
class FaceAligner:
    def __init__(self, gpu=0, target_folder="examples/output"):
        
        self.target_folder = target_folder
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        
        self.check = FaceCheck()
        warmupImg = cv2.imread('./DB/mim0001.jpg',)
        img_pil = Image.fromarray(warmupImg)
        self.check.setup_network(dummy_im=img_pil, dataset_setup=True, pre_recog=img_pil)

    def __call__(self, img, landmark, save, name="output.jpg"):
        return self.align(landmark, img, save, output=os.path.join(self.target_folder, name))

    def recognize(self, img, id)->str:
        img_rec = Image.fromarray(img)
        name, confidence = self.check.identify(img_rec, 0)
        print(f"{id}-> {name}({confidence:.2f})")
        return name

    def align(self, landmark, img, save=True,  output = "output.jpg"):
        output = os.path.join(self.target_folder, output)
        std_points_256 = np.array(
            [
                [85.82991, 85.7792],
                [169.0532, 84.3381],
                [127.574, 137.0006],
                [90.6964, 174.7014],
                [167.3069, 173.3733],
            ]
        )
        trans = transform.SimilarityTransform()
        #print("landmark:", landmark)
        res = trans.estimate(landmark, std_points_256)
        M = trans.params
        new_img = cv2.warpAffine(img, M[:2, :], dsize=(256, 256))
        img_rec = Image.fromarray(new_img).copy()
        new_img = new_img[:, :, ::-1]
        name, confidence = self.check.identify(img_rec, 0)
        # convert from cv2 BGR to RGB
        print(f"name:{name} {confidence:.2f}")
        
        if save:
            if name!= '':
                output = "examples/output/" + name + ".jpg"
            print("Saving to", output)
            cv2.imwrite(output, new_img)
        return name, confidence, new_img
    
    def save(self, img, name):
        output = os.path.join(self.target_folder, name)
        cv2.imwrite(output, img)