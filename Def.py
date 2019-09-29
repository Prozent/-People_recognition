import cv2
import numpy as np
import dlib
from typing import List, Optional, Tuple
from Struc import *
import os
import pickle


def module(a)->Tuple[float]:
    if a <0:
        a = a*-1
    return a

def poisk_person(frame)->Tuple[numpy.ndarray]:
    man_frame=[]
    predictor = dlib.shape_predictor("models/shape_predictor_5_face_landmarks.dat") #type:dlib.shape_predictor
    detector = dlib.get_frontal_face_detector() #type:dlib.fhog_object_detector
    squares = detector(frame) #type:List[dlib.rectangles[dlib.rectangle[(int, int) (int, int)]]]
    for face in squares:
        landmarks = predictor(frame, face) #type:dlib.full_object_detection
        f_a_l=[face,landmarks] #type:List[dlib.rectangle,dlib.full_object_detection]
        man_frame.append(f_a_l)
    return man_frame



def poisk_trak(traks, square)->Tuple[List[Trak], dlib.rectangle]:
    x1_train = square.left()
    y1_train = square.top()
    x2_train = square.right()
    y2_train = square.bottom()
    for trak in traks:
        kords=trak.square
        x3_val = kords.left()
        y3_val = kords.top()
        x4_val = kords.right()
        y4_val = kords.bottom()
        dis1 = (((x3_val - x1_train) ** 2) + ((y3_val - y1_train) ** 2))**0.5
        dis2 = (((x4_val - x2_train) ** 2) + ((y4_val - y2_train) ** 2))**0.5
        print('dist 1',dis1,' dis 2', dis2)
        if dis1+dis2<70  :
            return trak
    return None

def load_persons():
	p_files = os.listdir('/home/kirill/my-project-env/person/')  #type: List[str]
	p_names = list()  #type: List[str]
	p_descriptors = list()  #type: List[numpy.array]
	os.chdir('person')
	for p_file in p_files:
		name = os.path.splitext(p_file)[0]
		try:
			with open( p_file, "rb") as f:
				descriptor = pickle.load(f)
		except Exception as err:
			print(f"Failed to load person file, sckipping it [{p_file}][{err}]")
			continue
		#print(f"Loaded {name} - {len(descriptor)}")
		p_names.append(name)
		p_descriptors.append(numpy.array(descriptor))
	os.chdir('..')
	return p_names, p_descriptors

def is_landmark_ok(landmark):
    vec = np.empty([5, 2], dtype=int)
    for b in range(5):
        vec[b][0] = landmark.part(b).x
        vec[b][1] = landmark.part(b).y
    """distance_levi_1g = (((vec[2][0] - vec[1][0]) ** 2) + ((vec[2][1] - vec[1][1]) ** 2)) ** 0.5
    distance_prav_g = (((vec[4][0] - vec[3][0]) ** 2) + ((vec[4][1] - vec[3][1]) ** 2)) ** 0.5
    #print(distance_1g, ' distance glaz  ', distance_2g)
    print(vec)
    """
    matrix = [[vec[0][0] - vec[4][0], vec[0][1] - vec[4][1]], [vec[1][0] - vec[4][0], vec[1][1] - vec[4][1]]] #type:list[[int,int],[int,int]]
    ploshad = 0.5 * (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0])
    matrix1 = [[vec[2][0] - vec[4][0], vec[2][1] - vec[4][1]], [vec[3][0] - vec[4][0], vec[3][1] - vec[4][1]]]
    ploshad1 = 0.5 * (matrix1[0][0] * matrix1[1][1] - matrix1[0][1] * matrix1[1][0])
    print(ploshad, '   ', ploshad1)
    print(landmark, 'tut parts')
    if module(module(ploshad)-module(ploshad1))>50:
        print('False')
        return False
    else:
        print('True')
        return True
def poisk_id(baza_danih,diskriptor, min_disntace):

    descriptor = numpy.array(diskriptor)
    p_descriptors=baza_danih[1]
    distance = numpy.linalg.norm(p_descriptors - descriptor, axis=1)
    mi = int(numpy.argmin(distance))
    if distance[mi] > min_disntace:
        return None
    else:
        return mi