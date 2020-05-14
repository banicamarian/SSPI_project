import cv2
import math
import numpy as np

# Euclidian distance between 2 points
def L2_norm(x1, y1, x2, y2):
    #print("L2_norm", np.sqrt((x1 - x2)**2 + (y1 - y2)**2))
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# The top of the clock hand
def maxim_distance(center, clock_hand):
    if L2_norm(center[0], center[1], clock_hand[0], clock_hand[1]) > L2_norm(center[0], center[1], clock_hand[2], clock_hand[3]):
        return clock_hand[0], clock_hand[1]
    else:
        return clock_hand[2], clock_hand[3]

# Using atan2 get angle between 2 points(in our case, the horizontal axis and clock hand)
def get_angle_between_points(x_orig, y_orig, x_hand, y_hand):
    delta_x = x_hand - x_orig
    delta_y = y_hand - y_orig
    return (math.atan2(delta_y, delta_x) * 180) / math.pi

# After atan2 we get the angles in circle in 2 ways
#
#             _.-""""-._
#           .'          `.
#          /              \
#    -180°|                | 0°
# --------|----------------|-----------
#    +180°|                | 0°
#          \              /
#           `._        _.'
#              `-....-'
# This how we normalize the angles after this function
# it will result a simple formula for hours and minutes
#             360° | 0°
#             _.-"""""-._
#           .'           `.
#          /               \
#    +270°|                 | 90°
# --------|-----------------|-----------
#    +270°|                 | 90°
#          \               /
#           `._         _.'
#              `-.....-'
#             180° | 180°
def angle_to_hours(clock_center, point):
    angle = get_angle_between_points(clock_center[0], clock_center[1], point[0], point[1])
    if angle <= 0 and angle >= -90:
        angle = 90 + angle
    elif angle < -90 and angle >= -180:
        angle = 450 + angle
    elif angle > 0:
        angle = 90 + angle
    return angle

# Compute the minutes(using rule of three)
def minutes(angle):
    return (angle * 5) // 30

# Compute the hours(using rule of three)
# We also assure that when we are close to the edge of a new hour
# we get the correct value
def hours(angle, min):
    h = angle / 30
    if (math.ceil(h) < h + 0.5) and min < 30:
        return math.ceil(h)
    else:
        return math.floor(h)


if __name__ == '__main__':
    print(cv2.__version__)

    path = 'clock_0.png'

    orig_img = cv2.imread(path)
    # Open clock image and convert to grayscale(with 1 channel)
    clock_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Open clock image and convert it to grayscale(with 3 channels)
    clock_img = cv2.imread(path, cv2.COLOR_BGR2GRAY)

    # Resize image (width, height)
    dim = (800, 800)
    orig_img   =  cv2.resize(orig_img, dim, interpolation=cv2.INTER_AREA)
    clock_gray =  cv2.resize(clock_gray, dim, interpolation=cv2.INTER_AREA)
    clock_img  =  cv2.resize(clock_img, dim, interpolation=cv2.INTER_AREA)

    # Get HoughLines(using probabilistic transform)
    # Get edges in image
    edges = cv2.Canny(clock_img, 100, 150)
    # View edges from image
    cv2.imshow("Image edges", edges)

    min_line_length = 700
    min_line_gap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 90, min_line_length, min_line_gap)

    # Get HoughCircles to isolate the center of the clock
    circles = cv2.HoughCircles(clock_gray, cv2.HOUGH_GRADIENT, 1, 30,
                               param1=100, param2=140, minRadius=330, maxRadius=0)
    circles = np.uint16(np.around(circles))
    center = []
    for i in circles[0, :]:
        center = [i[0], i[1]]
        #print(i[0], i[1], i[2])
        # Draw outer circle
        # (image_name, center_circle, radius, color, thickness)
        cv2.circle(orig_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(orig_img, (i[0], i[1]), 2, (0, 0, 255), 2)

    print("Image center {}".format(center))

    # We set the parameters to get 4 lines
    print("Number of lines {}".format(len(lines)))
    line_length = {}
    number_of_line = 0
    for line in lines:
        #print(line[0])
        x1, y1, x2, y2 = line[0]
        dist = L2_norm(x1, y1, x2, y2)
        line_length.update({number_of_line: dist})
        number_of_line = number_of_line + 1
        cv2.line(clock_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Sort lines ascending(by length)
    sorted_line = sorted(line_length.items(), key=lambda item: item[1])
    # Compute the mean for hour and minute hand
    hour_hand   = (lines[sorted_line[0][0]][0] + lines[sorted_line[1][0]][0]) // 2
    minute_hand = (lines[sorted_line[2][0]][0] + lines[sorted_line[3][0]][0]) // 2

    # Draw the computed hour and minute hand on original image
    cv2.line(orig_img, (hour_hand[0], hour_hand[1]), (hour_hand[2], hour_hand[3]), (0, 255, 255), 3)
    cv2.line(orig_img, (minute_hand[0], minute_hand[1]), (minute_hand[2], minute_hand[3]), (0, 0, 255), 3)

    # Get the farthest point on each clock hand
    hour_hand_max = maxim_distance(center, hour_hand)
    minute_hand_max = maxim_distance(center, minute_hand)

    # Calculate minutes and hours
    m = minutes(angle_to_hours(center, minute_hand_max))
    h = hours(angle_to_hours(center, hour_hand_max), m)

    print("Hour: {} minute: {}".format(h, m))

    # Display the image on the screen
    cv2.imshow("Analog Clock with initial lines", clock_img)
    # Put hour and minutes on the image and display it on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 780)
    fontScale = 1
    fontColor = (255, 0, 255)
    lineType = 3
    cv2.putText(orig_img, "H: {} M: {}".format(h, m),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    cv2.imshow("Original image with mean for clock hands", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
