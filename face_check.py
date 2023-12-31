#
# Copyright (c) 2021 Takeshi Yamazaki
# This software is released under the MIT License, see LICENSE.
#
# source: https://github.com/take5553/face-check/blob/master/src/facecheck.py

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from mysettings import MySettings
from PIL import Image
from imageResizer import ImageResizer
import inspect

nn_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(min_face_size=120,  device=nn_device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(nn_device)


class FaceCheck():
    def __init__(self):
        self._registered = []
        self.settings = MySettings()
        self._npy_dir = self.settings.save_dir.main_dir
        self._pics_dir = self.settings.save_dir.onepic_dir_fullpath
        os.makedirs(self._npy_dir, exist_ok=True)
        self._npy_filename = 'dataset.npy'
        self._list_filename = 'filelist.txt'


    def setup_network(self, dummy_im=None, dataset_setup=True, pre_recog=None):
        print('FaceCheck initializing...')
        if not (dummy_im is None):
            global mtcnn
            print('Start pre-detection')
            mtcnn.detect(dummy_im)
            print('End pre-detection')
        if dataset_setup:
            print('Start loading registered dataset')
            # Load dataset
            if not os.path.exists(os.path.join(self._npy_dir, self._npy_filename)) or not os.path.exists(os.path.join(self._npy_dir, self._list_filename)):
                print('Renewing dataset')
                print("dataset:",os.path.join(self._npy_dir, self._npy_filename))
                print("filelist:",os.path.join(self._npy_dir, self._list_filename))
                self.make_dataset()
                self._file_list = self._load_filename_list()
            else:
                file_list_stored = self._load_filename_list()
                file_list_current = self._get_file_list()
                if (file_list_stored == file_list_current):
                    self._file_list = file_list_stored            
                else:
                    print('Renewing dataset')
                    self.make_dataset()
                    self._file_list = self._load_filename_list()
            self._registered = self._load_dataset()
            print('End loading registered dataset')
            print('Dataset shape is ' + str(self._registered.shape))
        if not (pre_recog is None):
            print('Start pre-recognition')
            self.identify(pre_recog, 0.6)
            print('End pre-recognition')
            
            
    def detect(self, img):
        global mtcnn
        return mtcnn.detect(img)
         
    def identify(self, img, threshold=0):
        vec = self._get_vec(img)
        if vec is None:
            return '', 0
        result = self._cos_sim_vs2d(self._registered, vec)
        result_idx = result.argmax()
        if result[result_idx] >= threshold:
            return self._file_list[result_idx][:-8], result[result_idx]
        else:
            return '', 0
        
    def make_dataset(self):
        fullpath_list = self._get_fullpath_list()
        vecs = self._make_vec_set(fullpath_list)
        np.save(os.path.join(self._npy_dir, self._npy_filename), vecs)
        file_list_str = '\n'.join(self._get_file_list())
        with open(os.path.join(self._npy_dir, self._list_filename), 'w') as f:
            f.write(file_list_str)
                
    def _make_vec_set(self, fullpath_list):
        vecs = []
        for file_path in fullpath_list:
            print('Converting to vec: {}'.format(file_path))
            img = cv2.imread(file_path)
            img = self.normalize_img_size(img)
            #cv2.imwrite(str(file_path).replace(".","1."), img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vec_shape = self._get_vec(img)
            print(vec_shape)
            vecs.append(vec_shape)
        return np.stack(vecs)

    def normalize_img_size_by_path(self, file_path):
        img = cv2.imread(file_path)
        img = ImageResizer().resize_with_letterbox_and_center_crop(img)
        return img
    
    def normalize_img_size(self, img):
        #img = ImageResizer().resize_with_letterbox_and_center_crop(img)
        return img
    
    def _get_file_list(self):
        files = sorted(os.listdir(self._pics_dir))
        return [f for f in files if os.path.isfile(os.path.join(self._pics_dir, f))]
    
    
    def _get_fullpath_list(self):
        files = sorted(os.listdir(self._pics_dir))
        return [os.path.join(self._pics_dir, f) for f in files if os.path.isfile(os.path.join(self._pics_dir, f))]
    
    
    def _cos_sim_vs2d(self, arr, vec):
        den = np.sqrt(np.einsum('ij,ij->i',arr,arr)*np.einsum('j,j',vec,vec))
        out = arr.dot(vec) / den
        return out


    def _get_vec(self, img):
        global mtcnn
        global resnet
        global nn_device
        img_cropped = mtcnn(img)
        if img_cropped == None:
            return None
        elif type(img_cropped) is torch.Tensor:
            img_embedding = resnet(img_cropped.unsqueeze(0).to(nn_device))
            return img_embedding.squeeze().to('cpu').detach().numpy().copy()
        else:
            img_cropped = torch.stack(img_cropped)
            img_embedding = resnet(img_cropped.to(nn_device))
            return img_embedding.to('cpu').detach().numpy().copy()
    
    
    def _load_dataset(self):
        return np.load(os.path.join(self._npy_dir, self._npy_filename))


    def _load_filename_list(self):
        with open(os.path.join(self._npy_dir, self._list_filename)) as f:
            filename_list = f.read()
        return str.splitlines(filename_list)
    
    
if __name__ == "__main__":
    check = FaceCheck()
    img = cv2.imread('./DB/mim0001.jpg',)
    img_pil = Image.fromarray(img)
    check.setup_network(dummy_im=img_pil, dataset_setup=True, pre_recog=img_pil)
    # for each image in folder examples/output
    # do the following
    path = "examples/output"
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        print ("File:",file)
        img = cv2.imread(filepath)
        if (img is None):
            print("No valid image")
            continue
        #img = ImageResizer().resize_with_letterbox_and_center_crop(img)
        img_pil = Image.fromarray(img)
        name, prob = check.identify(img_pil, 0.3)
        print(name, prob)
