import numpy as np
import cv2
from faceIdentity import FaceIdentity

def track_faces(mot_tracker, faces, known_ids):
    faceBboxes = [face[0] for face in faces]
    new_ids = []
    tracked_faceBboxes = mot_tracker.update(np.array(faceBboxes))

    # check if track[4] contains a new id
    for i,track in enumerate(reversed(tracked_faceBboxes)):
        id = int(track[4])
        face = faces[i]
        ids = [x for x in known_ids]
        if id not in ids:
            known_ids[id] = FaceIdentity(id,faceBox=face)
            new_ids.append((id, face))
            print("Added new identity:", known_ids[id])
    return tracked_faceBboxes, known_ids, new_ids

def draw_faces(img, tracks, known_ids):
    for track in tracks:
        bbox = track[:4]
        id = int(track[4])
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        identity = known_ids[id]
        name = identity.getName()
        img = cv2.putText(img, f"{str(id)}:{name}", (x1 +10 , y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def get_face_by_trackbb(img, track):
    bbox = track[:4]
    id = int(track[4])
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    return img[y1:y2, x1:x2]