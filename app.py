import tkinter as tk
# from sign_language_build import*
import customtkinter as ck
from PIL import Image, ImageTk 
from operator import truediv
from turtle import circle, color
import cv2
import numpy as np
import os
import pickle 
from matplotlib import pyplot as plt
import time
import datetime
import mediapipe as mp
from mediapineline import*
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

probList=np.array(["","","","","","","","","","","","","","","","","","","","",""],dtype=np.dtype('U100'))
window = tk.Tk()
window.geometry("900x600")
window.title("Alphabet Sign Language Detection")
ck.set_appearance_mode("dark")
labelProb= ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
labelProb.place(x=630, y=10)
labelProb.configure(text='PROBABILITY') 

labelA= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelA.place(x=630, y=50)
labelA.configure(text='A:')
labelAvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelAvalue.place(x=670, y=50)
labelAvalue.configure(text=probList[0])

labelB= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelB.place(x=760, y=50)
labelB.configure(text='B:')
labelBvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelBvalue.place(x=800, y=50)
labelBvalue.configure(text=probList[1])

labelC= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelC.place(x=630, y=90)
labelC.configure(text='C:')
labelCvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelCvalue.place(x=670, y=90)
labelCvalue.configure(text=probList[2])

labelD= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelD.place(x=760, y=90)
labelD.configure(text='D:')
labelDvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelDvalue.place(x=800, y=90)
labelDvalue.configure(text=probList[3])

labelE= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelE.place(x=630, y=130)
labelE.configure(text='E:')
labelEvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelEvalue.place(x=670, y=130)
labelEvalue.configure(text=probList[4])

labelF= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelF.place(x=760, y=130)
labelF.configure(text='F:')
labelFvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelFvalue.place(x=800, y=130)
labelFvalue.configure(text=probList[5])

labelG= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelG.place(x=630, y=170)
labelG.configure(text='G:')
labelGvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelGvalue.place(x=670, y=170)
labelGvalue.configure(text=probList[6])

labelH= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelH.place(x=760, y=170)
labelH.configure(text='H:')
labelHvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelHvalue.place(x=800, y=170)
labelHvalue.configure(text=probList[7])

labelI= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelI.place(x=630, y=210)
labelI.configure(text='I:')
labelIvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelIvalue.place(x=670, y=210)
labelIvalue.configure(text=probList[8])

labelJ= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelJ.place(x=760, y=210)
labelJ.configure(text='J:')
labelJvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelJvalue.place(x=800, y=210)
labelJvalue.configure(text=probList[9])

labelK= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelK.place(x=630, y=240)
labelK.configure(text='K:')
labelKvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelKvalue.place(x=670, y=240)
labelKvalue.configure(text=probList[10])

labelL= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelL.place(x=760, y=240)
labelL.configure(text='L:')
labelLvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelLvalue.place(x=800, y=240)
labelLvalue.configure(text=probList[11])

labelM= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelM.place(x=630, y=280)
labelM.configure(text='M:')
labelMvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelMvalue.place(x=670, y=280)
labelMvalue.configure(text=probList[12])

labelN= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelN.place(x=760, y=280)
labelN.configure(text='N:')
labelNvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelNvalue.place(x=800, y=280)
labelNvalue.configure(text=probList[13])

labelO= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelO.place(x=630, y=320)
labelO.configure(text='O:')
labelOvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelOvalue.place(x=670, y=320)
labelOvalue.configure(text=probList[14])

labelP= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelP.place(x=760, y=320)
labelP.configure(text='P:')
labelPvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelPvalue.place(x=800, y=320)
labelPvalue.configure(text=probList[15])

labelQ= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelQ.place(x=630, y=360)
labelQ.configure(text='Q:')
labelQvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelQvalue.place(x=670, y=360)
labelQvalue.configure(text=probList[16])

labelR= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelR.place(x=760, y=360)
labelR.configure(text='R:')
labelRvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelRvalue.place(x=800, y=360)
labelRvalue.configure(text=probList[17])

labelS= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelS.place(x=630, y=400)
labelS.configure(text='S:')
labelSvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelSvalue.place(x=670, y=400)
labelSvalue.configure(text=probList[18])

labelT= ck.CTkLabel(window, height=40, width=50, text_font=("Arial", 20), text_color="black", padx=10)
labelT.place(x=760, y=400)
labelT.configure(text='T:')
labelTvalue= ck.CTkLabel(window, height=40, width=80, text_font=("Arial", 20), text_color="red", padx=10)
labelTvalue.place(x=800, y=400)
labelTvalue.configure(text=probList[19])



def changeProb(probList):
    labelAvalue.configure(text=probList[0]+'%')
    labelBvalue.configure(text=probList[1]+'%')
    labelCvalue.configure(text=probList[2]+'%')
    labelDvalue.configure(text=probList[3]+'%')
    labelEvalue.configure(text=probList[4]+'%')
    labelFvalue.configure(text=probList[5]+'%')
    labelGvalue.configure(text=probList[6]+'%')
    labelHvalue.configure(text=probList[7]+'%')
    labelIvalue.configure(text=probList[8]+'%')
    labelJvalue.configure(text=probList[9]+'%')
    labelKvalue.configure(text=probList[10]+'%')
    labelLvalue.configure(text=probList[11]+'%')
    labelMvalue.configure(text=probList[12]+'%')
    labelNvalue.configure(text=probList[13]+'%')
    labelOvalue.configure(text=probList[14]+'%')
    labelPvalue.configure(text=probList[15]+'%')
    labelQvalue.configure(text=probList[16]+'%')
    labelRvalue.configure(text=probList[17]+'%')
    labelSvalue.configure(text=probList[18]+'%')
    labelTvalue.configure(text=probList[19]+'%')

labelDetection= ck.CTkLabel(window, height=100, width=200, text_font=("Arial", 100), text_color="blue", padx=10)
labelDetection.place(x=650, y=450)
labelDetection.configure(text='')

frame = tk.Frame(height=600, width=600)
frame.place(x=10, y=10) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) 
actions = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
'Q', 'R', 'S', 'T', '-'])


#Loading model
model = Sequential()
#model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('20letters.h5')
cap = cv2.VideoCapture(0)
# def do_detection():
#     ret, frame = cap.read()
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
#     results = pose.process(image)
#     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
#         mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5), 
#         mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10))
#     img = image[:, :460, :] 
#     imgarr = Image.fromarray(img)
#     imgtk = ImageTk.PhotoImage(imgarr) 
#     lmain.imgtk = imgtk 
#     lmain.configure(image=imgtk)
#     lmain.after(10, do_detection)
sequence = []
sentence = []
threshold = 0.9
predictions = []
holistic= mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def do_detection():
    global sequence
    global sentence
    global threshold
    global predictions
    global probList
    #Read Feed
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    #Make dectection
    image,results=mediapipe_detection(frame, holistic)
    #print(results)
    # Draw landmark
    draw_styled_landmarks(image,results)
    keypoints = extract_keypoints(results)

    sequence.append(keypoints)
    sequence = sequence[-30:]
    
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        #print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))
        
    #3. Viz logic
        if np.unique(predictions[-10:])[0]==np.argmax(res):
            if res[np.argmax(res)] > threshold: 
                
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])

                else:
                    sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # Viz probabilities
        for num, prob in enumerate(res):
            probList[num]=round(prob*100,1)
        changeProb(probList)
        
        labelDetection.configure(text=sentence[-1])
    
    img = image[:, :600, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(1, do_detection)
 #   for num, prob in enumerate(res)

do_detection()
window.mainloop()
