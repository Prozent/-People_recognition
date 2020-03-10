import dlib
import cv2
import pickle
import os
from scipy.spatial import distance
import numpy as np
sp = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

os.chdir('face')
name_face=os.listdir()

data_asd = []

def person_format(asd):
  text2=[]
  def raz(x):
    asd1=[]
    for i in range(len(x)):
      pere = [x[i][::-1]]
      asd1.append(pere)
    return asd1
  text1=raz(asd)
  ful_str=[]
  for j in text1:
    for i in np.arange(int(j[0].index('.')),(int(len(j[0])))):
      text2.append(j[0][i])
    full_data = [''.join(text2)]
    ful_str.append('nosrep'+full_data[0])
    text2=[]
  return raz(ful_str)
name_format=person_format(name_face)
print(name_face)
caunter = 0
for i in name_face:

    pic = cv2.imread(i)
    print(i)
    faces_1 = detector(pic, 1)
    shape = sp(pic, faces_1[0])
    face_descriptor_1 = facerec.compute_face_descriptor(pic, shape)
    data_asd.append(face_descriptor_1)
    os.chdir('../person')
    with open(name_format[caunter][0], 'wb') as f:
        pickle.dump(face_descriptor_1, f)
    caunter = caunter + 1

    os.chdir('../face')
