import cv2
import numpy as np
from pupil_apriltags import Detector

# Camera Matrix
K = np.array([
    [598.37142736,   0.0, 313.64827666],
    [0.0, 599.59958846, 237.87380463],
    [0.0,   0.0,   1.0]
], dtype=np.float64)

# Distortion coefficients
D = np.array([-7.75405759e-02, 4.69597542e-01, 6.68921205e-04,  -6.78959708e-04,  -9.65654022e-01])

tagSize = 0.05  # april tag size (meters)
half = tagSize / 2.0

# Tag corner coordinates in tag frame
# (consistent with pupil_apriltags corner order)
object_points = np.array([
    [-half,  -half, 0.0],
    [ half,  -half, 0.0],
    [ half, half, 0.0],
    [-half, half, 0.0]
], dtype=np.float64)

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1
)

# Video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not found")

print("AprilTag targeting running. Press ESC to quit.")

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)

    for det in detections:
        image_points = det.corners.astype(np.float64)

        # ---------------------------
        # Solve PnP (6-DOF pose)
        # ---------------------------
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            D,
            flags=cv2.SOLVEPNP_ITERATIVE
            # flags=cv2.SOLVEPNP_IPPE_SQUARE

        )

        if not success:
            continue

        # Translation (meters, camera frame)
        x, y, z = tvec.flatten()

        # Rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # ---------------------------
        # Euler angles (ZYX convention)
        # ---------------------------
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            roll  = np.arctan2(R[2, 1], R[2, 2])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll  = np.arctan2(-R[1, 2], R[1, 1])
            pitch = np.arctan2(-R[2, 0], sy)
            yaw   = 0.0

        roll, pitch, yaw = np.degrees([roll, pitch, yaw])

        # ---------------------------
        # Draw tag outline
        # ---------------------------
        corners = image_points.astype(int)
        for i in range(4):
            cv2.line(
                frame,
                tuple(corners[i]),
                tuple(corners[(i + 1) % 4]),
                (0, 255, 0),
                2
            )

        # ---------------------------
        # Draw coordinate axes
        # ---------------------------
        axis_len = tagSize / 2.0
        axes_3d = np.float64([
            [axis_len, 0, 0],  # X
            [0, axis_len, 0],  # Y
            [0, 0, axis_len]   # Z
        ])

        imgpts, _ = cv2.projectPoints(
            axes_3d, rvec, tvec, K, D
        )

        origin = tuple(corners.mean(axis=0).astype(int))

        cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)   # X (red)
        cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)   # Y (green)
        cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)   # Z (blue)

        # ---------------------------
        # Display pose info
        # ---------------------------
        pose_text = (
            f"x={x:.2f}m y={y:.2f}m z={z:.2f}m\n"
            f"r={roll:.1f} p={pitch:.1f} y={yaw:.1f}"
        )

        y0 = origin[1] - 80
        for i, line in enumerate(pose_text.split("\n")):
            cv2.putText(
                frame,
                line,
                (origin[0] - 80, y0 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        print(
            f"ID {det.tag_id} | "
            f"x={x:.3f} y={y:.3f} z={z:.3f} | "
            f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}"
        )

    cv2.imshow("AprilTag Targeting (6-DOF)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
11
cap.release()
cv2.destroyAllWindows()
