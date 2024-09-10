import cv2 as cv
import numpy as np

# Constants
KNOWN_DISTANCE = 45  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 3.0  # INCHES

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
FONTS = cv.FONT_HERSHEY_COMPLEX

yoloNet_path = r"Obstacle Detection/yolov4-tiny.cfg"
yoloNet2_path = r"Obstacle Detection/yolov4-tiny.weights"

# Load YOLO model
yoloNet = cv.dnn.readNet(yoloNet2_path, yoloNet_path)
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Function to detect objects in an image
def object_detector(image):
    classes, scores, boxes = model.detect(
        image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = GREEN
        label = f"{classid} : {score}"
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
        data_list.append([classid, box[2], (box[0], box[1]-2)])
    return data_list

# Function to find focal length
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Function to calculate distance
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance * 0.0833  # Convert inches to feet

# Read reference images
ref_person = cv.imread(r'Obstacle Detection/ReferenceImages/H.png')
ref_mobile = cv.imread(r'Obstacle Detection/ReferenceImages/M2.png')

# Verify image dimensions
if ref_person is None or ref_mobile is None:
    print("Error: Could not read reference images.")
    exit()

person_data = object_detector(ref_person)
if len(person_data) == 0:
    print("Error: No objects detected in the person reference image.")
    exit()
person_width_in_rf = person_data[0][1]

mobile_data = object_detector(ref_mobile)
if len(mobile_data) == 0:
    print("Error: No objects detected in the mobile reference image.")
    exit()
mobile_width_in_rf = mobile_data[0][1]

print(f"Person width in pixels: {person_width_in_rf}, Mobile width in pixels: {mobile_width_in_rf}")

# Find focal lengths
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

# Initialize video capture
cap = cv.VideoCapture(0)

# Main loop for object detection and distance estimation
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from the camera.")
        break

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
        else:
            continue

        x, y = d[2]
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Distance: {round(distance, 2)} ft', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
