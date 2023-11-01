import cv2

class ImageResizer:
    def __init__(self, size=160):
        self.size = size

    def resize_with_letterbox(self, img):
        height, width = img.shape[:2]
        max_dim = max(height, width)
        scale = self.size / max_dim
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height))
        top = (self.size - new_height) // 2
        bottom = self.size - new_height - top
        left = (self.size - new_width) // 2
        right = self.size - new_width - left
        letterboxed_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return letterboxed_img
    
    def center_crop(self,img):
        height, width = img.shape[:2]
        if height > width:
            top = 0
            bottom = height - width
            left = 0
            right = 0
        else:
            top = 0
            bottom = 0
            left = 0
            right = width - height
        cropped_img = img[top:height-bottom, left:width-right]
        return cropped_img
    
    def resize_with_letterbox_and_center_crop(self, img):
        height, width = img.shape[:2]
        max_dim = max(height, width)
        scale = self.size / max_dim
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized_img = cv2.resize(img, (new_width, new_height))
        top = (self.size - new_height) // 2
        bottom = self.size - new_height - top
        left = (self.size - new_width) // 2
        right = self.size - new_width - left
        letterboxed_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cropped_img = self.center_crop(letterboxed_img)
        return cropped_img