### importing required libraries
import gc
from json.encoder import INFINITY
from socket import timeout
import torch
import os
import cv2
from time import time, sleep
import win32api, win32con
import pyautogui
import numpy as np
import keyboard
import mss
from math import sqrt
import PySimpleGUI as sg
import serial
from ctypes import *
import math

aimbot = True # Enables aimbot if True

arduinoMode = False # Using an arduino mouse spoof?

screenShotWidth = 320 # Width of the detection box
screenShotHeight = 320 # Height of the detection box

headshot_mode = True # Pulls aim up towards head if True

no_headshot_multiplier = 0.2 # Amount multiplier aim pulls up if headshot mode is false
headshot_multiplier = 0.35 # Amount multiplier aim pulls up if headshot mode is true

detection_threshold = 0.65 # Cutoff enemy certainty percentage for aiming

lockKey = 0x14

sct = mss.mss()

layout = [
    [
        sg.Text("Model Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        [sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )],
        [sg.Text('Game Window', size=(15, 1)), sg.InputText("Counter", key="gw1")],
        [sg.Text('Lock Distance', size=(15, 1)), sg.InputText("100", key="ld1")],
        [sg.Text('Lock Speed', size=(15, 1)), sg.InputText("2", key="ls1")],
        [sg.Button('Start'), sg.Button('Exit')]
    ],
]

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
    frame = [frame]
    #print(f"[INFO] Detecting. . . ")
    results = model(frame)
    
    # results.show()
    # print( results.xyxyn[0])
    # print(results.xyxyn[0][:, -1])
    # print(results.xyxyn[0][:, :-1])

    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates

def FindPoint(x1, y1, x2,
              y2, x, y) :
    if (x > x1 and x < x2 and
        y > y1 and y < y2) :
        return True
    else :
        return False

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, area, arduino, lockDistance, lockSpeed, classes):

    """
    --> This function takes results, frame and classes
    --> results: contains labels and coordinates predicted by model on the given frame
    --> classes: contains the strting labels

    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    #print(f"[INFO] Total {n} detections. . . ")
    #print(f"[INFO] Looping through all detections. . . ")

    best_detection = None

    closest_mouse_dist = INFINITY
    
    cWidth = area["width"] / 2
    cHeight = area["height"] / 2

    ### looping through to find closest target to mouse
    for i in range(n):
        row = cord[i]
        if row[4] >= detection_threshold: ### threshold value for detection. We are discarding everything below this value
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates

            ### Check dist to mouse and if closest select this
            centerx = x1 - (0.5*(x1-x2))
            centery = y1 - (0.5*(y1-y2))

            centerx = centerx - cWidth
            centery = centery - cHeight
            
            dist = sqrt((0-centerx)**2 + (0-centery)**2)
            
            if dist < closest_mouse_dist and classes[int(labels[i])] == 'enemy' or 'person' or '0' or 'body' and dist < lockDistance:
                best_detection = row
                closest_mouse_dist = dist

            # Draw bbox for this detection    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox   
            

    if best_detection is not None:
        x1, y1, x2, y2 = int(best_detection[0]*x_shape), int(best_detection[1]*y_shape), int(best_detection[2]*x_shape), int(best_detection[3]*y_shape) ## BBOx coordniates

        box_height = y1 - y2

        if headshot_mode == True:
            headshot_offset = box_height * headshot_multiplier
        else:
            headshot_offset = box_height * no_headshot_multiplier    
                
        centerx = x1 - (0.5*(x1-x2))
        centery = y1 - (0.5*(y1-y2))

        centerx = centerx - cWidth
        centery = (centery + headshot_offset) - cHeight

        if aimbot == True and win32api.GetKeyState(lockKey) and arduinoMode == False:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(centerx * lockSpeed), int(centery * lockSpeed), 0, 0)
        elif aimbot == True and win32api.GetKeyState(lockKey) and arduinoMode == True:
            centerx = centerx - 960
            centery = centery - 540
            arduino.write(((centerx * lockSpeed) + ':' + (centery * lockSpeed) + 'x').encode())

    #print(f"[INFO] Finished extraction, returning frame!")
    return frame

### ---------------------------------------------- Main function -----------------------------------------------------
def main(arduino=False, run_loop=False, modelPath=None, gameWindow=None, lockSpeed=None, lockDist=None):

    if arduino == True:
        arduino = serial.Serial('COM5', 9600, timeout=1)

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model = torch.hub.load('./yolov5', 'custom', source='local', path=modelPath, force_reload=True)

    classes = model.names ### class names in string format

    if run_loop==True:

        # Selecting the correct game window
        try:
            videoGameWindows = pyautogui.getWindowsWithTitle(gameWindow)
            videoGameWindow = videoGameWindows[0]
        except:
            print("The game window you are trying to select doesn't exist.")
            print("Check variable videoGameWindowTitle (typically on line 36)")
            exit()

        # Select that Window
        videoGameWindow.activate()

        sctArea = {"mon": 1, "top": videoGameWindow.top + (videoGameWindow.height - screenShotHeight) // 2,
                         "left": ((videoGameWindow.left + videoGameWindow.right) // 2) - (screenShotWidth // 2),
                         "width": screenShotWidth,
                         "height": screenShotHeight}

        #cv2.namedWindow("vid", cv2.WINDOW_NORMAL)

        count = 0
        sTime = time()
        
        print("Program Working!")

        while True:

            img = sct.grab(sctArea)

            img = np.array(img)

            frame = img

            #print(f"[INFO] Working with frame {frame_no} ")
            if (gameWindow == "Halo Infinite"):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            results = detectx(frame, model = model)          
            frame = plot_boxes(results, frame, sctArea, arduino, lockDistance=lockDist, lockSpeed=lockSpeed, classes=classes)                
                
            #cv2.imshow("vid", frame)

            if cv2.waitKey(1) and 0xFF == ord('q'):
                break

            if keyboard.is_pressed('esc'):
                print(f"[INFO] Exiting. . . ")               
                break

            # Forced garbage cleanup every second
            count += 1
            if (time() - sTime) > 1:
                #print("CPS: {}".format(count))
                count = 0
                sTime = time()

                gc.collect(generation=0)

        print(f"[INFO] Cleaning up. . . ")
        
        ## closing all windows
        exit()  

def selectSettings():
    window = sg.Window("Proton Client", layout)

    chosenModel = "replaceme.pt"

    while True:
        event, values = window.read()

        # Folder name was filled in, make a list of files in the folder
        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                # Get list of files in folder
                file_list = os.listdir(folder)
            except:
                file_list = []

            fnames = [
                f
                for f in file_list
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith((".pt"))
            ]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-":  # A file was chosen from the listbox
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                chosenModel = filename
            except:
                pass
        
        elif event == 'Start':
            if values['ld1'] != "":
                ld = float(values['ld1'])
            else:
                ld = 100

            if values['gw1'] != "":
                gw = values['gw1']
            else:
                gw = "Counter"

            if values['ls1'] != "":
                ls = float(values['ls1'])
            else:
                ls = 2
            break

        elif event == "Exit" or event == sg.WIN_CLOSED:
            window.close()
            exit()

    window.close()

    print("Model Path: ", str(chosenModel))
    print("Lock Distance: ", str(ld))
    print("Lock Speed: ", str(ls))
    print("Game Window: ", str(gw))

    return chosenModel, gw, ld, ls

### -------------------  calling the main function-------------------------------

chosenModel, gw, ld, ls = selectSettings()

main(run_loop=True, modelPath=chosenModel, gameWindow=gw, lockDist=ld, lockSpeed=ls)


