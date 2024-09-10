import cv2 as cv
import numpy as np
import playsound
from gtts import gTTS
import time
import random
import os

# Distance constants
CAMERA_FOCAL_LENGTH = 500  # in feet

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255),
          (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []

y1 = r'Distance Estimator/yolov4-tiny.weights'
y2 = r'Distance Estimator/yolov4-tiny.cfg'
p1 = r'Distance Estimator/classes.txt'
p2 = r'Distance Estimator/classes_width.txt'
p3 = r'Distance Estimator/ignored_classes.txt'
with open(p1, "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
# setttng up opencv net
yoloNet = cv.dnn.readNet(y1, y2)

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

# object detector function /method


def object_detector(image, model):

    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(classid) % len(COLORS)]
        # label = "%s : %f" % (class_names[classid[0]], score)
        label = "%s : %f" % (class_names[classid], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

        data_list.append(
            [class_names[classid], box[2], (box[0], box[1]-2)])
    return data_list


def speak(text):
    tts = gTTS(text=text)
    filename = "voice.mp3"
    tts.save(filename)
    audio = AudioSegment.from_file(filename)
    audio = audio.set_frame_rate(16000)
    audio.export(filename, format="mp3")
    playsound(filename)
    os.remove(filename)


def distance_finder(real_object_width, width_in_frmae):
    distance = (real_object_width * CAMERA_FOCAL_LENGTH) / width_in_frmae
    return distance  # edit


def to_ignore(obj_class):
    with open(p3) as f:
        lines = f.readlines()
        if obj_class in lines:
            return True
    return False


def get_object_width(obj_class):
    with open(p2) as f:
        lines = f.readlines()
        matching = [s for s in lines if obj_class in s]
        # print(matching)
        index = lines.index(matching[0])
        width = lines[index].split('-')[1]
    return width


def get_pos(frame_x, frame):
    h, w, channels = frame.shape
    half = w//2
    if frame_x <= half:
        return 'LEFT'
    return 'RIGHT'


def detect_obj_in_frame(data, frame):
    right_count = 0
    left_count = 0
    for d in data:
        obj_class = d[0]
        if to_ignore(obj_class) == False:
            # width is in inches convert it to feet
            width_ft = int(get_object_width(obj_class).strip())*0.0833333
            distance = distance_finder(int(width_ft), d[1])
            x, y = d[2]
            if distance < 5:
                # winsound.Beep(440, 500)
                # check if the object is on the right or left of the image
                position = get_pos(x, frame)
                if position == 'RIGHT':
                    right_count += 1
                elif position == 'LEFT':
                    left_count += 1
            cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} ft',
                       (x+5, y+13), FONTS, 0.48, GREEN, 2)
            cv.imshow('frame', frame)
    return left_count, right_count


cap = cv.VideoCapture(0)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

while True:
    ret, frame_read = cap.read()
    frame = cv.cvtColor(frame_read, cv.COLOR_BGR2RGB)
    data = object_detector(frame, model)
    left_count, right_count = detect_obj_in_frame(data, frame)
    seen_left = 0
    seen_right = 0
    # check for conditions
    if right_count > 0 and left_count > 0:
        if seen_left > 3 and seen_right > 3:
            speak(
                'Please wait for a few seconds for the obstacle to clear')
            time.sleep(2)
        elif seen_left <= 3 and seen_right <= 3:
            speak_message = 'Please see left' if random.randint(
                0, 1) == 0 else 'Please see right'
            print('Speak message:', speak_message)
            time.sleep(2)
        elif seen_left > 3:
            print('Speak message: Please see right')
            time.sleep(2)
        elif seen_right > 3:
            print('Speak message: Please see left')
            time.sleep(2)
            seen_right += 1
        elif right_count > 0 and left_count == 0:
            if seen_left > 0 and seen_left <= 3:
                speak(f'Please turn left {seen_left} times')
                # turn left those many times as seen left
                time.sleep(2)
            elif seen_right > 0 and seen_right <= 3:
                speak(f'Please turn right {seen_right} times')
                time.sleep(2)
            # again turn left
            speak('Please turn left')
            time.sleep(2)
            seen_left = 0
            seen_right = 0
        elif left_count > 0 and right_count == 0:
            if seen_left > 0 and seen_left <= 3:
                # turn left those many times as seen left
                speak(f'Please turn left {seen_left} times')
                time.sleep(2)
            elif seen_right > 0 and seen_right <= 3:
                speak(f'Please turn right {seen_right} times')
                time.sleep(2)
            # again turn right
            speak('Please turn right')
            time.sleep(2)
            seen_left = 0
            seen_right = 0

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()