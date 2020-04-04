import numpy as np
import cv2
import time
import Person

import os
import pathlib

import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from enum import Enum

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

######################
## TENSORFLOW SETUP ##
######################

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
#PATH_TO_LABELS = '/content/drive/My Drive/Capstone/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

print(detection_model.inputs)
detection_model.output_dtypes
detection_model.output_shapes

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  #print(output_dict['detection_classes'])
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))

def calc_person(image_path):
  image_np = np.array(Image.open(image_path))
  # Actual detection.
  output_dict = run_inference_for_single_image(detection_model, image_np)
  potential_persons = []
  for i in range(output_dict['detection_classes'].size):
    print(output_dict['detection_classes'][i])
    if output_dict['detection_classes'][i] == 1:
      potential_persons.append(i)
  bestScore = 0
  for i in range(len(potential_persons)):
    if output_dict['detection_scores'][potential_persons[i]] > bestScore:
      bestScore = output_dict['detection_scores'][potential_persons[i]]
  return bestScore



############
## OPENCV ##
############

cap = cv2.VideoCapture(sys.argv[1]) #Open video file
cv2.namedWindow('Frame')

font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 60
pid = 1


fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) #Create the background subtractor
kernelOp = np.ones((5,5),np.uint8)
kernelCl = np.ones((7,7),np.uint8)
areaMin = 100
areaMax = 10000
distanceMargin = 30


# Crosswalk bounds
endpoint1_start_point = (0, 0)
endpoint1_end_point = (0, 0)
endpoint1_dimensions = [0, 0, 0, 0] # centerX, centerY, width, height
endpoint2_start_point = (0, 0)
endpoint2_end_point = (0, 0)
endpoint2_dimensions = [0, 0, 0, 0]
crosswalk_start_point = (0, 0)
crosswalk_end_point = (0, 0)
crosswalk_dimensions = [0, 0, 0, 0]

people_in_crosswalk = 0
people_in_endpoints = 0

class DrawingMode(Enum):
    Endpoint1 = 1
    Endpoint2 = 2
    Crosswalk = 3
    NotDrawing = 4

drawing_mode = DrawingMode.NotDrawing


def start_processing():
    global is_paused
    global cap, font, persons, max_p_age, pid, fgbg, kernelOp, kernelCl, areaMin, areaMax, distanceMargin
    global endpoint1_start_point, endpoint1_end_point, endpoint1_dimensions
    global endpoint2_start_point, endpoint2_end_point, endpoint2_dimensions
    global crosswalk_start_point, crosswalk_end_point, crosswalk_dimensions
    global people_in_crosswalk, people_in_endpoints
    global number_pedestrians_label, number_pedestrians_in_crosswalk_label, intersection_status_label

    iter = 0
    while(cap.isOpened()):
        iter = iter + 1
        if is_paused:
            time.sleep(1)
        else:
            ret, frame = cap.read() #read a frame

            imgHeight, imgWidth, channels = frame.shape 

            fgmask = fgbg.apply(frame) #Use the substractor
            try:
                ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
                # Opening (erode->dilate)
                mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
                #C losing (dilate -> erode)
                mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
            except:
                #if there are no more frames to show...
                print('EOF')
                break

            contours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in contours0:
        #        cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
                area = cv2.contourArea(cnt)
                print("area " + str(area))
                if area > areaMin and area < areaMax:
                    #################
                    #   TRACKING    #
                    #################            
                    M = cv2.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)
              
                    new = True
                    for i in persons:
                      if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)
                        # Update confidence every 60 frames
                        if iter == 60:
                            top = y - 20
                            bottom = y + h + 20
                            left = x - 20
                            right = x + w + 20

                            if top < 0:
                              top = 0
                            if bottom >= imgHeight:
                              bottom = imgHeight
                            if left < 0:
                              left = 0
                            if right >= imgWidth:
                              right = imgWidth
                            cv2.imwrite("person.jpg", frame[top:bottom, left:right])
                            p.updateConfidence(calc_person("person.jpg"))
                        break
                    if new == True:
                        p = Person.MyPerson(pid,cx,cy, max_p_age)
                        persons.append(p)
                        pid += 1     

                        top = y - 20
                        bottom = y + h + 20
                        left = x - 20
                        right = x + w + 20

                        if top < 0:
                          top = 0
                        if bottom >= imgHeight:
                          bottom = imgHeight
                        if left < 0:
                          left = 0
                        if right >= imgWidth:
                          right = imgWidth
                        cv2.imwrite("person.jpg", frame[top:bottom, left:right])
                        p.updateConfidence(calc_person("person.jpg"))
                        #print(p.getConfidence())
                                
                  
                    #cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)            
                    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.drawContours(frame, cnt, -1, (0,255,0), 3)

            #Age all persons and delete timed out
            for i in range(len(persons) - 1, -1, -1):
                if persons[i].timedOut() == True:
                    print('timedout' + str (persons[i].getId()))
                    print(len(persons))
                    del persons[i]
                    print(len(persons))
                else:
                    persons[i].age_one()

            print(persons)


            people_in_endpoints = 0
            people_in_crosswalk = 0
            for i in persons:
                print('person ' + str (i.getId()) + ' age ' + str(i.getAge()) + ' timedout? ' + str(i.getDone()))
                if i.getConfidence() > 0.3:
                    if len(i.getTracks()) >= 2:
                        pts = np.array(i.getTracks(), np.int32)
                        pts = pts.reshape((-1,1,2))
                        frame = cv2.polylines(frame,[pts],False,i.getRGB())
                    cv2.putText(frame, (str(i.getId()) + " " + str(round(i.getConfidence()*100)) + "%"),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)

                    # Calculate people in intersection & endpoints
                    # Crosswalk has priority
                    dx = max(abs(i.getX() - crosswalk_dimensions[0]) - crosswalk_dimensions[2] / 2, 0);
                    dy = max(abs(i.getY() - crosswalk_dimensions[1]) - crosswalk_dimensions[3] / 2, 0);
                    dist = dx * dx + dy * dy;
                    if dist < distanceMargin:
                        people_in_crosswalk = people_in_crosswalk + 1
                        continue
                    # Check endpoint 1
                    dx = max(abs(i.getX() - endpoint1_dimensions[0]) - endpoint1_dimensions[2] / 2, 0);
                    dy = max(abs(i.getY() - endpoint1_dimensions[1]) - endpoint1_dimensions[3] / 2, 0);
                    dist = dx * dx + dy * dy;
                    if dist < distanceMargin:
                        people_in_endpoints = people_in_endpoints + 1
                        continue
                    dx = max(abs(i.getX() - endpoint2_dimensions[0]) - endpoint2_dimensions[2] / 2, 0);
                    dy = max(abs(i.getY() - endpoint2_dimensions[1]) - endpoint2_dimensions[3] / 2, 0);
                    dist = dx * dx + dy * dy;
                    if dist < distanceMargin:
                        people_in_endpoints = people_in_endpoints + 1
                        continue

            # Draw boxes for intersection items
            cv2.rectangle(frame,endpoint1_start_point,endpoint1_end_point,(255,0,255),1)
            cv2.rectangle(frame,endpoint2_start_point,endpoint2_end_point,(255,0,255),1)
            cv2.rectangle(frame,crosswalk_start_point,crosswalk_end_point,(0,0,255),1)

            number_pedestrians_label.setText('Pedestrians: ' + str(people_in_crosswalk + people_in_endpoints))
            if people_in_crosswalk > 0:
                intersection_status_label.setText('<font color="red">PEDESTRIANS DETECTED</font>')
            elif people_in_endpoints > 0:
                intersection_status_label.setText('<font color="orange">Possible pedestrians approaching</font>')
            else:
                intersection_status_label.setText('<font color="green">No Pedestrians detected</font>')
            number_pedestrians_in_crosswalk_label.setText('Pedestrians in Crosswalk: ' + str(people_in_crosswalk))

            cv2.imshow('Frame',frame)

            #Abort and exit with 'Q' or ESC
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        if iter == 60:
            iter = 0
        
        key = cv2.waitKey(1)
        if key == ord("p"):
            time.sleep(0.1)
            cv2.waitKey(0)

    cap.release() #release video file
    cv2.destroyAllWindows() #close all openCV windows



##############
## PYQT GUI ##
##############

first_run = True
is_paused = True

def pause_button_pressed():
    global is_paused, first_run
    global endpoint1_start_point, endpoint1_end_point, endpoint1_dimensions
    global endpoint2_start_point, endpoint2_end_point, endpoint2_dimensions
    global crosswalk_start_point, crosswalk_end_point, crosswalk_dimensions

    print('Pause Button Pressed')

    # Check that all bounding boxes are set
    if endpoint1_start_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0
    if endpoint1_end_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0
    if endpoint2_start_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0
    if endpoint2_end_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0
    if crosswalk_start_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0
    if crosswalk_end_point == (0, 0):
        intersection_status_label.setText('<font color="red">WARNING: All bounds not set</font>')
        return 0

    endpoint1_dimensions[0] = (endpoint1_start_point[0] + endpoint1_end_point[0]) / 2
    endpoint1_dimensions[1] = (endpoint1_start_point[1] + endpoint1_end_point[1]) / 2
    endpoint1_dimensions[2] = endpoint1_end_point[0] - endpoint1_start_point[0]
    endpoint1_dimensions[3] = endpoint1_end_point[1] - endpoint1_start_point[1]

    endpoint2_dimensions[0] = (endpoint2_start_point[0] + endpoint2_end_point[0]) / 2
    endpoint2_dimensions[1] = (endpoint2_start_point[1] + endpoint2_end_point[1]) / 2
    endpoint2_dimensions[2] = endpoint2_end_point[0] - endpoint2_start_point[0]
    endpoint2_dimensions[3] = endpoint2_end_point[1] - endpoint2_start_point[1]

    crosswalk_dimensions[0] = (crosswalk_start_point[0] + crosswalk_end_point[0]) / 2
    crosswalk_dimensions[1] = (crosswalk_start_point[1] + crosswalk_end_point[1]) / 2
    crosswalk_dimensions[2] = crosswalk_end_point[0] - crosswalk_start_point[0]
    crosswalk_dimensions[3] = crosswalk_end_point[1] - crosswalk_start_point[1]

    is_paused = not is_paused
    if first_run == True:
        first_run = False
        start_processing()

def draw_crosswalk_button_pressed():
    global is_paused, drawing_mode

    print('Draw Crosswalk Button Pressed')

    if is_paused:
        drawing_mode = DrawingMode.Crosswalk

def draw_endpoint1_button_pressed():
    global is_paused, drawing_mode

    print('Draw First Endpoint Button Pressed')

    if is_paused:
        drawing_mode = DrawingMode.Endpoint1

def draw_endpoint2_button_pressed():
    global is_paused, drawing_mode

    print('Draw Second Endpoint Button Pressed')

    if is_paused:
        drawing_mode = DrawingMode.Endpoint2

def clear_button_pressed():
    global is_paused, drawing_mode
    global firstFrame
    global endpoint1_start_point, endpoint1_end_point
    global endpoint2_start_point, endpoint2_end_point
    global crosswalk_start_point, crosswalk_end_point

    print('Clear Button Pressed')

    firstFrame = startingFrame.copy()

    cv2.imshow('Frame',firstFrame)
    cv2.waitKey(0)

    # Reset rects
    endpoint1_start_point = (0, 0)
    endpoint1_end_point = (0, 0)
    endpoint2_start_point = (0, 0)
    endpoint2_end_point = (0, 0)
    crosswalk_start_point = (0, 0)
    crosswalk_end_point = (0, 0)

def exit_button_pressed():
    global is_paused
    is_paused = True
    cv2.destroyAllWindows()
    sys.exit()

app = QApplication([])
win = QMainWindow()
central_widget = QWidget()
pause_button = QPushButton('Play/Pause', central_widget)
pause_button.clicked.connect(pause_button_pressed)
draw_crosswalk_button = QPushButton('Draw Crosswalk', central_widget)
draw_crosswalk_button.clicked.connect(draw_crosswalk_button_pressed)
draw_endpoint1_button = QPushButton('Draw First Endpoint', central_widget)
draw_endpoint1_button.clicked.connect(draw_endpoint1_button_pressed)
draw_endpoint2_button = QPushButton('Draw Second Endpoint', central_widget)
draw_endpoint2_button.clicked.connect(draw_endpoint2_button_pressed)
clear_button = QPushButton('Clear', central_widget)
clear_button.clicked.connect(clear_button_pressed)
number_pedestrians_label = QLabel()
number_pedestrians_label.setText('Pedestrians: 0')
intersection_status_label = QLabel()
number_pedestrians_in_crosswalk_label = QLabel()
number_pedestrians_in_crosswalk_label.setText('Pedestrians in Crosswalk: 0')
exit_button = QPushButton('Exit', central_widget)
exit_button.clicked.connect(exit_button_pressed)
layout = QVBoxLayout(central_widget)
layout.addWidget(pause_button)
layout.addWidget(draw_crosswalk_button)
layout.addWidget(draw_endpoint1_button)
layout.addWidget(draw_endpoint2_button)
layout.addWidget(clear_button)
layout.addWidget(number_pedestrians_label)
layout.addWidget(intersection_status_label)
layout.addWidget(number_pedestrians_in_crosswalk_label)
layout.addWidget(exit_button)
win.setCentralWidget(central_widget)
win.show()



# Drawing tools
drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_mouse(event,x,y,flags,param):
    global ix,iy,drawing,firstFrame
    global drawing_mode
    global endpoint1_start_point, endpoint1_end_point
    global endpoint2_start_point, endpoint2_end_point
    global crosswalk_start_point, crosswalk_end_point

    if drawing_mode != DrawingMode.NotDrawing:

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        # elif event == cv2.EVENT_MOUSEMOVE:
        #     if drawing == True:
        #         cv2.rectangle(firstFrame,(ix,iy),(x,y),(0,255,0),2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # startpoint normalizing
            if ix > x:
                temp = x
                x = ix 
                ix = temp
            if iy > y:
                temp = y
                y = iy 
                iy = temp
            if drawing_mode == DrawingMode.Endpoint1 or drawing_mode == DrawingMode.Endpoint2:
                cv2.rectangle(firstFrame,(ix,iy),(x,y),(255,0,255),1)
            elif drawing_mode == DrawingMode.Crosswalk:
                cv2.rectangle(firstFrame,(ix,iy),(x,y),(0,0,255),1)
            if drawing_mode == DrawingMode.Endpoint1:
                endpoint1_start_point = (ix, iy)
                endpoint1_end_point = (x, y)
            elif drawing_mode == DrawingMode.Endpoint2:
                endpoint2_start_point = (ix, iy)
                endpoint2_end_point = (x, y)
            elif drawing_mode == DrawingMode.Crosswalk:
                crosswalk_start_point = (ix, iy)
                crosswalk_end_point = (x, y)
            drawing_mode = DrawingMode.NotDrawing
            cv2.imshow('Frame',firstFrame)
            cv2.waitKey(0)


# Show first frame to draw
ret, startingFrame = cap.read() #read a frame

app.exit(app.exec_())
