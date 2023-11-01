from batch_face import drawLandmark_multiple

def draw_landmarks(img, faces, pts, regressor=None):
    if regressor is not None:
        boxes = [face[0] for face in faces]
        pts = regressor(boxes, img)
    for face, landmarks in zip(faces, pts):
        img = drawLandmark_multiple(img, face[0], landmarks)
    return img