def detect_faces(detector, img, faces, fps_target_counter):
    
    faces = detector(img, cv=True, threshold=0.7)
    fps_target_counter = 0
    #print("Counter hit!")
        
    return faces, fps_target_counter