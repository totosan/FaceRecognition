import face_recognition
import numpy as np
import os, io, sys, pickle
import cv2

def recognize_faces(known_face_encodings, known_face_names, face_image, face):
    #face_locations = face_recognition.face_locations(face_image, number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(face_image, face, num_jitters=20, model="large")
    print("\tFound {} faces in the image.".format(len(face_encodings)))
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print("\tface_distances:", face_distances)
        best_match_index = np.argmin(face_distances)
        print("\tbest_match_index:", best_match_index)
        print("\tMatches:", matches)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    return face_names

def load_known_faces_from_files():
    # open folder and read all images
    known_face_encodings = []
    known_face_names = []
    
    for file in os.listdir("DB"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join("DB", file))
            print("\tLoading", file)
            location = face_recognition.face_locations(img)
            if len(location) > 0:
                known_face_encodings.append(
                    face_recognition.face_encodings(img,location, num_jitters=20, model="large")[0]
                    )
                known_face_names.append(file.split(".")[0])
            else:
                raise Exception("No face found in", file)
    return known_face_encodings, known_face_names

def load_known_faces_from_files_NoLoc():
    # open folder and read all images
    known_face_encodings = []
    known_face_names = []
    
    for file in os.listdir("DB"):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join("DB", file))
            print("\tLoading", file)
            known_face_encodings.append(
                face_recognition.face_encodings(img, num_jitters=20, model="large")[0]
                )
            known_face_names.append(file.split(".")[0])
    return known_face_encodings, known_face_names

def save_known_faces(known_face_encodings, known_face_names):
    # save encodings and names to file as database
    with open("known_faces.dat", "wb") as face_data_file:
        face_data = [known_face_encodings, known_face_names]
        pickle.dump(face_data, face_data_file)
        print("Known faces backed up to disk.")
def load_known_faces_from_DB():
    # load encodings and names to file as database
    with open("known_faces.dat", "rb") as face_data_file:
        face_data = pickle.load(face_data_file)
        print("Known faces loaded from disk.")
        return face_data[0], face_data[1]
    
if __name__ == "__main__":
    if True:
        print("Loading known faces...")
        encodings, names = load_known_faces_from_files_NoLoc()
        print(names)
        #save to disk
        save_known_faces(encodings, names)
    encodings = []
    names = []
    #load from disk
    encodings, names = load_known_faces_from_DB()
    #exit()
    # Load an image with an unknown face from examples/output
    for file in os.listdir("examples/output"):
        if file.endswith(".jpg") or file.endswith(".png"):
            print("Recognizing faces in", file)
            img = cv2.imread(os.path.join("examples/output/",file), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face = [(0, img.shape[1], img.shape[0],0)]
            recognized_name = recognize_faces(encodings, names, img, face)
            print(recognized_name)