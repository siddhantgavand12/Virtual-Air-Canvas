import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Giving diffrent arras to handle colours pointes of diffrent colour
'''The purpose of this line is to create a container (bpoints) that will be used to 
store points associated with the color blue. The use of deque with a maximum length 
can be helpful in scenarios where you want to maintain a limited history of points or 
implement a rolling buffer for real-time applications, such as drawing or tracking.*'''
bpoints = [deque(maxlen = 1024)] 
gpoints = [deque(maxlen = 1024)] 
rpoints = [deque(maxlen = 1024)] 
ypoints = [deque(maxlen = 1024)] 

#These indexes used to mark the points in particular arrays of specific color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

#The kernel to be used for dilation purposed
kernel = np.ones((5,5),np.uint8)

colors =[(255,0,0), (0,255,0), (0,0,255), (0,255,255)]
colorIndex = 0
cv2.namedWindow('Output', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow('Output', 800, 600)

# Here is code for White canvas setup
paintEWindow = np.zeros((471,636,3)) + 255
paintEWindow = cv2.rectangle(paintEWindow, (40,1), (140,65), (0,0,0), 2)
paintEWindow = cv2.rectangle(paintEWindow, (160,1), (255,65), (255,0,0), 2)
paintEWindow = cv2.rectangle(paintEWindow, (275,1), (370,65), (0,255,0), 2)
paintEWindow = cv2.rectangle(paintEWindow, (390,1), (485,65), (0,0,255), 2)
paintEWindow = cv2.rectangle(paintEWindow, (505,1), (600,65), (0,255,255), 2)

cv2.putText(paintEWindow, "CLEAR", (49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintEWindow, "BLUE", (185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintEWindow, "GREEN", (298,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintEWindow, "RED", (420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
cv2.putText(paintEWindow, "YELLOW", (520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
# cv2.namedWindow('Paint', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Paint', 800, 600)

cv2.namedWindow('Paint', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow('Paint', 800, 600)


# Ititialize mediapipe using the below two lines
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands =1 , min_detection_confidence = 0.7)
# This is used for drawing the landmarks on the detected hand
mpDraw = mp.solutions.drawing_utils

# Initializing the webcam 
cap = cv2.VideoCapture(0)
ret = True
while ret:
    ret , frame = cap.read()

    x, y, c = frame.shape

    # Read each frame vertically
    frame = cv2.flip(frame, 1)
    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)

    cv2.putText(frame, "CLEAR", (49,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "RED", (420,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame, "YELLOW", (520,33), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0),2,cv2.LINE_AA)
    # frame = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    #Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id,lm)
                # print(lm.x)
                # print(lm.y)
                #Adjust according to our frame size
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx,lmy])

            #Drawing landmarks on frames
            mpDraw.draw_landmarks(frame,handslms,mpHands.HAND_CONNECTIONS,mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))
        
       # Get the tip of the hand
        tip_of_hand = (landmarks[8][0], landmarks[8][1])

        # Determine the color based on the selected color
        selected_color = colors[colorIndex]
        color = (int(selected_color[0]), int(selected_color[1]), int(selected_color[2]))


        paintEWindow = np.zeros((471, 636, 3)) + 255
        # Draw a pointer on the paint window
        cv2.circle(paintEWindow, tip_of_hand, 10, color, 2) 
        cv2.circle(frame, tip_of_hand, 10, color, 2) 
    

        #here 8 means 8th finger and 0 & 1 means x and y corinates of of this finger
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
       
        print(center[1]-thumb[1])
        if(thumb[1]-center[1] < 30):
            bpoints.append(deque(maxlen = 512))
            blue_index += 1
            gpoints.append(deque(maxlen = 512))
            green_index += 1
            rpoints.append(deque(maxlen = 512))
            red_index += 1
            ypoints.append(deque(maxlen = 512))
            yellow_index += 1
        
        # Y cordinate
        elif center[1] <= 65:
            #Remove all points from dqueue
            if 40 <= center[0] <= 140: #clear button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintEWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # red
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # yellow

        else:
            
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
    # Append the next deques when nothing is detedctd to avoid messing up
    else:
        bpoints.append(deque(maxlen = 512))
        blue_index += 1
        gpoints.append(deque(maxlen = 512))
        green_index += 1
        rpoints.append(deque(maxlen = 512))
        red_index += 1
        ypoints.append(deque(maxlen = 512))
        yellow_index += 1


    #Draw lines of al the colors on the canvas and frame
    points = [bpoints,gpoints,rpoints,ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
             for k in range (1,len(points[i][j])):
                  # Checks if either of the points is None. If so, it skips drawing a line for that pair of points.
                  if points[i][j][k-1] is None or points[i][j][k] is None:
                       continue
                  cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                  cv2.line(paintEWindow,points[i][j][k-1],points[i][j][k],colors[i],2)


    cv2.imshow("Output", frame)   
     
    cv2.imshow("Paint",paintEWindow)

    if cv2.waitKey(1) == ord('q'):
         break
    
#release the webcam and destroy active windows
cap.release()
cv2.destroyAllWindows()

    
                
            











