import cv2
import os

cap = cv2.VideoCapture(0)

os.makedirs("Calibration Image Folder")
num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Calibration Capture", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        filename = f"{"Calibration Image Folder"}/calib_{num}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        num += 1
cap.release()
cv2.destroyAllWindows()