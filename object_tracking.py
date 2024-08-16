import cv2 as cv
import numpy as np

# Initialize video capture from the default camera
cap = cv.VideoCapture(0)

while True:
    # Capture each frame from the video feed
    _, frame = cap.read()

    # Convert the frame from BGR to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the HSV range for blue, green, and red colors
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for blue, green, and red colors
    mask_blue = cv.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv.inRange(hsv, lower_green, upper_green)
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)

    # Combine all the masks
    combined_mask = cv.bitwise_or(mask_blue, mask_green)
    combined_mask = cv.bitwise_or(combined_mask, mask_red)

    # Perform bitwise AND operation to keep only the colored parts of the image
    res = cv.bitwise_and(frame, frame, mask=combined_mask)

    # Find contours in the combined mask
    contours, _ = cv.findContours(combined_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:  # Filter out small contours by area
            # Get the bounding box coordinates for the contour
            x, y, w, h = cv.boundingRect(contour)
            # Draw a rectangle around the detected object
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with the bounding boxes
    cv.imshow('Frame', frame)
    # Display the combined mask
    cv.imshow('Mask', combined_mask)
    # Display the result of the bitwise AND operation
    cv.imshow('Result', res)

    # Exit the loop when the user presses the 'q' key
    if cv.waitKey(5) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv.destroyAllWindows()
