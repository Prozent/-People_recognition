import numpy
import cv2
from PIL import Image
import time
from Def import *
from Struc import *
MIN_DISTANCE=0.5 #minimum distance similar to the database
traks=[]
p_names, p_descriptors = load_persons()
baza_danih = [p_names, p_descriptors]
x=0 #Переменная для счётчика чтобы прервать бесконечный цикл

def ReadCamera(Camera):
    while True:
        cap = cv2.VideoCapture(Camera)
        (grabbed, frame) = cap.read()
        if grabbed == True:
            yield frame

while True:
    #xxx = input()
    #ret, frame = video_capture.read()
    for frame in ReadCamera(0):

       # cadr=frame
       break
    trak = Trak()
    t_l=poisk_person(frame)
    for iterat in t_l:
        trak=poisk_trak(traks,iterat[0]) #поиск траков плохо ведёться
        #print(trak)
        if trak is None:
            trak = Trak()
            trak.square=iterat[0]
            trak.landmarki=iterat[1]
        trak.square=iterat[0]
        image_sqr=np.array(frame)
        image_sqr = cv2.rectangle(image_sqr, (iterat[0].left(),iterat[0].top()),(iterat[0].right(),iterat[0].bottom()), (225,255,0))
        trak.time=time.time()
        if trak.id !=None:
           # print()
            #if trak.id is 0: #очень часто незнает, разобраться надо
             #   print('Mi ego ne znaem')
            #else:
             #   print(baza_danih[0][trak.id],'ntut ')
            continue
        if not is_landmark_ok(iterat[1]):
            continue
        facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')
        diskriptor = facerec.compute_face_descriptor(frame,iterat[1]) #type:dlib.vector
        #print(diskriptor)
        id=poisk_id(baza_danih,diskriptor,MIN_DISTANCE)

        if id is not None:
            print('Тут возможно : ', baza_danih[id][0])
            trak.id=id
        else:
            print('Тут возможно : тот кого мы незнаем')

            trak.id=0
        traks.append(trak)
    old_traks = []
    for i in reversed(range(len(traks))):

        if time.time()-traks[i].time>10:
            old_traks.append(i)
    #print(old_traks, 'tut index')
    #print(len(traks), 'Tut collicestco tacof do udalenii')
    if len(old_traks) !=0:
        for i in old_traks:
            del traks[i]
            continue
        #print(len(traks), 'Tut collicestco tacof posle delite')
    print(len(traks))

    cv2.imshow('Video', frame)
    cv2.imshow('Vide1o', image_sqr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#video_capture.release() закрывает видео с камеры