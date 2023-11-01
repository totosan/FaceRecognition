import time

from batch_face import RetinaFace, LandmarkPredictor, ShapeRegressor
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaConvertColor, cudaAllocMapped, cudaDeviceSynchronize, cudaFromNumpy, cudaMemcpy


from sort.sort import Sort
import numpy as np
from face_aligner import FaceAligner
from face_detection import detect_faces
from face_tracking import get_face_by_trackbb, track_faces, draw_faces
from face_landmarks import draw_landmarks
from faceIdentity import FaceIdentity
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for client to attach...")
debugpy.wait_for_client()

def initialize():
    predictor = LandmarkPredictor(gpu_id=0, backbone="PFLD", file=None)
    detector = RetinaFace(0)
    regressor = ShapeRegressor(gpu_id=0)
    mot_tracker = Sort()
    aligner = FaceAligner(gpu=0, target_folder="examples/output")
    return predictor, detector, mot_tracker, aligner, regressor

def capture_and_convert(input):
    img_cdnn = input.Capture()
    if img_cdnn is None:
        time.sleep(1)
        return None
    img = cudaToNumpy(img_cdnn)
    return img

def main():
    predictor, detector, mot_tracker, aligner, regressor = initialize()

    #input = videoSource("webrtc://@:8554/in", argv=None)
    input = videoSource("/dev/video0", argv=None)
    ouput = videoOutput("webrtc://@:8554/out", argv=None)
    faces = None
    detailedLandmarks = None
    fps_target_counter=0
    known_ids = {}
    
    while True:
        start = time.time()
        img = capture_and_convert(input)
        if img is None:
            continue
        
        if faces is None or fps_target_counter % 50 == 0:
            faces = detector(img, cv=True, threshold=0.7)
            fps_target_counter = 0
            
        if fps_target_counter > 0 and detailedLandmarks is not None and len(detailedLandmarks) > 0:
            faces_old = faces.copy()
            faces.clear()
            for face, result in zip(faces_old,detailedLandmarks):                
                ldm_new = result
                (x1, y1), (x2, y2) = ldm_new.min(0), ldm_new.max(0)
                box_new = np.array([x1, y1, x2, y2])
                box_new[:2] -= 10
                box_new[2:] += 10
                faces.append([box_new, face[1], None])
            
        
        if len(faces) == 0:
            print("NO face is detected!")
            face = None
            continue
        else:
            detailedLandmarks = predictor(faces, img, from_fd=True)
            
            # Draw the tracked faces on the image
            tracks, known_ids, new_ids = track_faces(mot_tracker, faces, known_ids)
            # return all ids from known_ids, that are also in tracks (track[4])
            displayed_ids = [id for id in known_ids if id in [int(track[4]) for track in tracks]]
            for id in displayed_ids:
                identity = FaceIdentity()
                identity = known_ids[id]
                face = identity.getFace()
                if (id in [i for (i,_) in new_ids]):
                    print("New ID:", id)
                    print("\tStarting timer for this face id")
                    face_timer = (time.time(), id)
                    name, confidence, aligned_img = aligner(img, identity.getFace()[1], True, f"output{id}.jpg")
                    print(f"\t\tFace recognized: {name} with id {id} and confidence {confidence:.2f}")
                    identity.startRecognition( lambda: identity.stopRecognition(), 5)
                    #identity.startRecognition(lambda: identity.stopRecognition(lambda: print("Stopped recognition id:",identity.getID())), 5)
                    identity.setName(name)
                    identity.setConfidence(confidence)
                    print("---------------------------------")
                # face is a tuple of (box, landmarks, face_image)
                if(identity.getFace()[1] is not None and identity.isInRecognition() and fps_target_counter % 10 == 0):
                        name, confidence, aligned_img = aligner(img, face[1], False, f"output{id}.jpg")
                        if(confidence > 0.6):
                            recognizedName = name
                        else:
                            recognizedName = f'{name}?'
                            
                        #known_ids[id] = FaceIdentity(id=id, name=recognizedName, faceBox=face)
                        known_ids[id].setConfidence(confidence)
                        known_ids[id].setName(recognizedName)
                        known_ids[id].setFace(face)
                        print(f"Face recognized: {recognizedName} with id {id}")

                        
                        
            if (False):                    
                #create opencv image with white background of same size as img
                imgWhite = np.zeros_like(img)
                imgWhite.fill(255)
                img = draw_faces(imgWhite, tracks)
                # Draw the facial landmarks on the image
                img = draw_landmarks(imgWhite, faces, detailedLandmarks, regressor=None)
            else:
                img = draw_faces(img, tracks, known_ids)
                # Draw the facial landmarks on the image
                #img = draw_landmarks(img, faces, detailedLandmarks, regressor=None)

        
        fps_target_counter +=1
        # convert to CUDA (cv2 images are numpy arrays, in BGR format)
        img_cdnn = cudaFromNumpy(img, isBGR=False)                
        
        ouput.Render(img_cdnn)
        ouput.SetStatus("FPS: {:.2f}".format(1 / (time.time() - start)))
        #print("FPS=", 1 / (time.time() - start))


if __name__ == "__main__":
        main()