#!/usr/bin/env python

"""
Modified from code posted here: http://forums.pimoroni.com/t/pan-tilt-hat-repo/3402/11
"""
import numpy as np
import cv2
import os
import sys
import time
import pantilthat as pth
import argparse


# os.system('sudo modprobe bcm2835-v4l2')


# Input Args/Switches
ap = argparse.ArgumentParser()
ap.add_argument("-bbb", "--show_blue_border_box", action="store_true", default=0, help="Draws a (blue) border boundary box which can dictate camera movement. Default: Show box")
ap.add_argument("-p", "--show_person_box", action="store_true", default=0, help="Draws a (green) box around the person. Default: Show box")
ap.add_argument("-m", "--still_camera", action="store_true", default=0, help="Keep camera Still when enabled (1); otherwise the camera moves (0). Default: Camera moves")
ap.add_argument("-c", "--is_cascade", action="store_true", default=0, help="Whether to use LBP Cascade (1) or Haar Cascade (0). Default: Haar Cascade (0)")
args = ap.parse_args()


# GLOBALS
key_letter = 27
show_text = False
is_yolo_face = False
show_boundary_box = args.show_blue_border_box
show_person_box = args.show_person_box
is_camera_still = args.still_camera
is_cascade = args.is_cascade
shrink_person_box = False
show_onscreen_help = False
# Frame dimensions vars
FRAME_W = FRAME_H = 0
x = y = w = h = 0
# Boundary Box vars
w_min = w_max = h_min = h_max = 0


def man_move_camera(key_press):
    """Take keystrokes to dictate camera movemement.

    # Argument:
        key_press: takes in one letter for movement
                    (same as gaming controls, no inversion):
                        w: Up
                        a: Left
                        s: Down
                        d: Right

    # Returns
        None
    """
    cam_pan = pth.get_pan()
    cam_tilt = pth.get_tilt()
    move_x = 0
    move_y = 0

    if(key_press.lower() == 'a'):
        move_x = -2
    elif(key_press.lower() == 'd'):
        move_x = 2
    elif(key_press.lower() == 's'):
        move_y = -1
    elif(key_press.lower() == 'w'):
        move_y = 1

    if((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        cam_pan += move_x
        pth.pan(int(cam_pan))
        time.sleep(0.005)
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        cam_tilt -= move_y
        pth.tilt(int(cam_tilt))
        time.sleep(0.005)
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    return


def move_camera(x, y, w, h):
    """Takes in object tracking coordinates and
        moves camera to try to "center" the subject.

    # Argument:
        x: coordinate on the x axis where subject is detected
        y: coordinate on the y axis where subject is detected
        w: width of object detected on screen
        h: height of object detected on screen

    # Returns
        None
    """
    if(is_camera_still):
        return
    if(shrink_person_box):
        #for camera tracking only
        # shrink height and width by half
        w = w//2
        h = h//2
        # center box by adding a quarter of w/h to x/y
        x += w//2
        y += h//2

    cam_pan = pth.get_pan()
    cam_tilt = pth.get_tilt()
    move_x = 2
    move_y = 1
    yolo_offset = 0 if is_cascade else (h_min * -0.75)

    if(((x + w)*0.95 > w_max) & (x*0.95 < w_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(w > (w_max - w_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(x + w/2 > (FRAME_W + w_min)/2):
            cam_pan += move_x
            pth.pan(int(cam_pan))
        elif(x - w/2 < (FRAME_W - w_min)/2):
            cam_pan -= move_x
            pth.pan(int(cam_pan))
    elif((cam_pan + move_x < 90) & (cam_pan - move_x > -90)):
        if(x + w > w_max):
            cam_pan += move_x
            pth.pan(int(cam_pan))
        elif(x < w_min):
            cam_pan -= move_x
            pth.pan(int(cam_pan))
    else:
        print(f'MAX PAN - cannot move:  {cam_pan + move_x}')

    if(((y + h)*0.95 > h_max) & (y*0.95 < h_min)):
        # If both subject borders take up 95% or
        # more of the boundary box, do nothing
        pass
    elif(h > (h_max - h_min)*0.95):
        # If subject border-length take up 95% (not centered)
        # or more of the boundary box, correct movement by aligning centers
        if(y + h/2 > (FRAME_H + h_min)/2):
            cam_tilt += move_y
            pth.tilt(int(cam_tilt))
        elif(y - h/2 < (FRAME_H - h_min)/2):
            cam_tilt -= move_y
            pth.tilt(int(cam_tilt))
    elif((cam_tilt + move_y < 90) & (cam_tilt - move_y > -90)):
        if(y + h > h_max):
            cam_tilt += move_y
            pth.tilt(int(cam_tilt))
        elif(y < h_min + yolo_offset):
            cam_tilt -= move_y
            pth.tilt(int(cam_tilt))
    else:
        print(f'MAX TILT - cannot move:  {cam_tilt + move_y}')
    return


def reset_camera_position():
    """Resets Camera position.

    # Argument:
        None

    # Returns:
        None
    """
    pth.pan(0)
    pth.tilt(-20)
    time.sleep(2)


cascade_path = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml' 
if(is_cascade):
    cascade_path = '/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)


## Lights removed at the moment ##
# light_mode(WS2812)

# def lights(r,g,b,w):
#     for x in range(18):
#         set_pixel_rgbw(x,r if x in [3,4] else 0,g if x in [3,4] else 0,b,w if x in [0,1,6,7] else 0)
#     show()

# lights(0,0,0,50)

reset_camera_position()


cap = cv2.VideoCapture(0)
    
# Set placement vars
FRAME_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # width
FRAME_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # height
# For BOUNDARY BOX
w_min = int((FRAME_W)/6)
w_max = int((FRAME_W) - w_min)
h_min = int((FRAME_H)/5)
h_max = int((FRAME_H) - h_min)


while(True):
    ret, frame = cap.read()
    if(not ret):
        print("Error getting image")
        continue

    frame = cv2.flip(frame, 0)
   
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 

    if(True):
        # t = cv2.GetTickCount()
        # HaarDetectObjects takes 0.02s
        # faces = cv2.HaarDetectObjects(small_img, cascade, cv2.CreateMemStorage(0), haar_scale, min_neighbors, haar_flags, min_size)
        faces = face_cascade.detectMultiScale(frame, 1.1, 3)
        # t = cv2.GetTickCount() - t
        if(True):
        # if faces:


            # lights(50 if len(faces) == 0 else 0, 50 if len(faces) > 0 else 0,0,50)

            # for ((x, y, w, h), n) in faces:
            for (x, y, w, h) in faces:
                # # the input to cv2.HaarDetectObjects was resized, so scale the
                # # bounding box of each face and convert it to two CvPoints
                # pt1 = (int(x * image_scale), int(y * image_scale))
                # pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                # cv2.Rectangle(frame, pt1, pt2, cv2.RGB(100, 220, 255), 1, 8, 0)
                # # get the xy corner co-ords, calc the midFace location
                # x1 = pt1[0]
                # x2 = pt2[0]
                # y1 = pt1[1]
                # y2 = pt2[1]

                # midFaceX = x1+((x2-x1)/2)
                # midFaceY = y1+((y2-y1)/2)
                # midFace = (midFaceX, midFaceY)

                # offsetX = midFaceX / float(frame.width/2)
                # offsetY = midFaceY / float(frame.height/2)
                # offsetX -= 1
                # offsetY -= 1

                # cam_pan -= (offsetX * 5)
                # cam_tilt += (offsetY * 5)
                # cam_pan = max(0,min(180,cam_pan))
                # cam_tilt = max(0,min(180,cam_tilt))

                # print(offsetX, offsetY, midFace, cam_pan, cam_tilt, frame.width, frame.height)
                
                
                if(show_person_box):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Haar Cascade', (x, y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 1,
                                cv2.LINE_AA)

                # pth.pan(int(cam_pan-90))
                # pth.tilt(int(cam_tilt-90))
                if(not is_camera_still):
                    move_camera(x, y, w, h)
                break
    
    if(show_boundary_box):
        cv2.rectangle(frame, (w_min, h_min), (w_max, h_max), (255, 0, 0), 2)
                
    # Display the resulting frame
    cv2.imshow('Tracker', frame)
    
    key_stroke = cv2.waitKey(1)
    key_letter = key_stroke
    if key_stroke & 0xFF == ord('q'):
        break
    elif key_stroke & 0xFF == 27:
        break
    elif key_stroke & 0xFF == ord('b'):
        show_boundary_box = not show_boundary_box
    elif key_stroke & 0xFF == ord('p'):
        show_person_box = not show_person_box
    elif key_stroke & 0xFF == ord('m'):
        is_camera_still = (not is_camera_still)
    elif key_stroke & 0xFF == ord('c'):
        is_cascade = not is_cascade
        
    if key_stroke & 0xFF == ord('r'):
        reset_camera_position()
    elif key_stroke & 0xFF == ord('w'):
        man_move_camera('w')
    elif key_stroke & 0xFF == ord('a'):
        man_move_camera('a')
    elif key_stroke & 0xFF == ord('s'):
        man_move_camera('s')
    elif key_stroke & 0xFF == ord('d'):
        man_move_camera('d')


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
