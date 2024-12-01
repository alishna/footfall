import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import time


model=YOLO('yolov8s.pt')
# model = YOLO('runs/detect/train7/weights/best.pt')
# model = YOLO('runs/detect/train18/weights/best.pt')

# area1=[(290, 448), (595, 576),(506, 593), (265,471)]
# area2=[(336, 394), (626, 472),(580,546), (317,444)]
area1=[(268, 472), (325, 467), (567, 583), (482, 593)]
area2=[(343, 474), (574, 578), (604, 534), (434, 442)]



people_enter={}
# Define the RGB callback function to get the color on mouse move
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        color = frame[y, x]  # Access the pixel color in BGR format
        # print(f"BGR Color at ({x}, {y}): {color}")
coordinates = []
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates.append((x, y))  # Append the coordinates of the click
        print(f"Clicked at: ({x}, {y})")  # Print the coordinates on the console
        # Draw a circle on the clicked point to visualize the click
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Video", frame)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

n_p = 0

#cap=cv2.VideoCapture(r'peoplecount1/peoplecount1.mp4')
#cap=cv2.VideoCapture("D:\loonibha\code\peoplecounteryolov8\shoppingmall.mp4","r")
cap = cv2.VideoCapture(r"E:\django\yolov8-students-counting-lobby\peoplecount1.mp4")

# Read the class names for YOLO detections
my_file = open("E:\django\yolov8-students-counting-lobby\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count=0
people_entering={}
people_exiting={}
tracker = Tracker()
entering = set()
exiting = set()

while True:    
    ret, frame = cap.read()
    if not ret:
        print("no ret")
        break
    if frame is None:
        print("Error: Received empty frame.")
        break
    frame=cv2.resize(frame,(1200,600))

    # print(f"Frame shape: {frame.shape}")  
    count += 1
    if count % 2 != 0:
        continue
    # frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list = []
    conf_threshold = 0.5
    for index, row in px.iterrows():
        x1, y1, x2, y2, conf = row[:5]
        if conf < conf_threshold:  # Skip low-confidence detections
            continue
        print(f"Drawing box: {x1}, {y1}, {x2}, {y2}, Confidence: {conf}")  # Debug
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw the box

    for index, row in px.iterrows():
        x1, y1, x2, y2, d = map(int, row[:5])
        c = class_list[d]
        if 'person' in c:
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, c, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            list.append([x1, y1, x2, y2])

   
    # for bbox in bbox_id:
    #     x3, y3, x4, y4, id = bbox
    #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
    #     # Check for exiting the first area (area2)
    #     results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
    #     if results >= 0 and id not in people_exiting and id not in people_entering:
    #         people_exiting[id] = (x4, y4)  # Mark as exiting
    #         cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Draw a red rectangle around the person

    #     # Check for exiting the second area (area1)
    #     if id in people_exiting:
    #         results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
    #         if results1 >= 0 and id not in exiting:  # Make sure it's only added once
    #             exiting.add(id)
    #             cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
    #             # cv2.circle(frame, (x4, y4), 4, (255, 0, 255), 1)
    #             # cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    #     # Check for entering the first area (area1)
    #     results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
    #     if results2 >= 0 and id not in people_entering and id not in people_exiting:
    #         people_entering[id] = (x4, y4)  # Mark as entering
    #         cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)  # Draw blue rectangle around the person

    #     # Check for entering the second area (area2)
    #     if id in people_entering:
    #         results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
    #         if results3 >= 0 and id not in exiting:  # Make sure it's only added once
    #             exiting.add(id)
    #             cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
    #             cv2.circle(frame, (x4, y4), 4, (255, 0, 255), 1)
    #             cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    # # Draw the areas (area1 and area2)
    # cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    # cv2.putText(frame, str('1'), (410, 531), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    # cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # # Initialize the number of people if not already initialized
    # if 'number' not in locals():
    #     number = 0

    # # Calculate the number of people entering and exiting
    # i = len(people_entering)
    # o = len(exiting)

    # # Update the number of people inside
    # number += i  # Increment for entering
    # number -= o  # Decrement for exiting

    # # Display the number of people entering
    # cv2.putText(frame, "Entering: " + str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)

    # # Display the number of people exiting
    # cv2.putText(frame, "Exiting: " + str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

    # # Display the total number of people inside
    # cv2.putText(frame, "No of people: " + str(number), (100, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)


    bbox_id = tracker.update(list)
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        

        # Check for exiting the first area (area2)
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0 and id not in people_exiting:
            people_exiting[id] = (x4, y4)  # Mark as exiting
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2) # Draw a red rectangle around the person

        # Check for exiting the second area (area1)
        if id in people_exiting:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0 and id not in exiting:  # Make sure it's only added once
                exiting.add(id)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                # cv2.circle(frame, (x4, y4), 4, (255, 0, 255), 1)
                # cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # Check for entering the first area (area1)
        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0 and id not in people_entering:
            people_entering[id] = (x4, y4)  # Mark as entering
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2) #draw blue

        # Check for entering the second area (area2)
        if id in people_entering:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3 >= 0 and id not in entering:  # Make sure it's only added once
                entering.add(id)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), 1)
                cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('1'), (410, 531), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, str('2'), (466, 485), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    # Initialize number if not already initialized
    if 'number' not in locals():
        number = 0
        
    if 'number1' not in locals():
        number1 = 0

    i = len(entering)
    o = len(exiting)

    if i:
        number += 1
    if o:
        number1 += 1

    # Display the number of people entering
    cv2.putText(frame, "Entering: " + str(i), (60, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    # Display the number of people exiting
    cv2.putText(frame, "Exiting: " + str(o), (60, 140), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

    # Display the total number of people
    # cv2.putText(frame, "No of people: " + str(number), (100, 200), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)


    # Display the frame with bounding boxes and count
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    

cap.release()
cv2.destroyAllWindows()


