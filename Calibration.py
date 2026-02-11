import cv2
import numpy as np
import glob

import os
print("Working directory:", os.getcwd())


# Chessboard settings
chessboard_size = (8, 5)  # inner corners
square_size = 0.009  # meters

# Prepare object points
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob("./Calibration Images/calib_*.jpg")
print("Found images:", images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\nCamera Matrix:")
print(camera_matrix)

print("\nDistortion:")
print(dist_coeffs)
