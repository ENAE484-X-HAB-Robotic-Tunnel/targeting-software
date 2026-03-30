import cv2
import numpy as np
from pupil_apriltags import Detector

def CameraReadings():
    #Inputs: None
    #Outputs: AprilTag measurements of IDs 0,1,2
    # Camera Matrix
    K = np.array([
        [579.22798778,    0.0,         308.93921177],
        [0.0,    581.28267983,         234.87942809],
        [0.0,    0.0,         1.0]
    ], dtype=np.float64)

    # Distortion coefficients
    D = np.array([
        [-0.03920679, 0.24817876, 0.00089033, -0.00258005, -0.53929358]
    ], dtype=np.float64)

    detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1
    )
    tagSize = 0.05

    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    print("AprilTag targeting running. Press Space to capture.")

    # Initialize dictionary
    # Key: Tag ID
    # Value: [x,y,z,r,p,y]
    tag_measurements = {} 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(
    gray,
    estimate_tag_pose=True,
    camera_params=(K[0,0], K[1,1], K[0,2], K[1,2]),
    tag_size=tagSize
)
        x = y = z = roll = pitch = yaw = None
        for det in detections:
            tag_id = det.tag_id
            if tag_id not in [0,1,2]:
                continue
            image_points = det.corners.astype(np.float64)

            #R_cam and tvec are obtained directly
            R_cam = det.pose_R
            tvec = det.pose_t

            # Translation (meters, camera frame)
            x_cam, y_cam, z_cam = tvec.flatten()
            x = x_cam; y = y_cam; z = z_cam
            # Translate to SP frame 
            x = z_cam; y = -x_cam; z = -y_cam

            # Rotation matrix
            T = np.array([
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]
            ])
            
            # Translate to SP frame
            R_SP = T @ R_cam @ T.T
            
            # ---------------------------
            # Euler angles (ZYX convention)
            # ---------------------------
            sy = np.sqrt(R_SP[0, 0]**2 + R_SP[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                roll  = np.arctan2(R_SP[2, 1], R_SP[2, 2])
                pitch = np.arctan2(-R_SP[2, 0], sy)
                yaw   = np.arctan2(R_SP[1, 0], R_SP[0, 0])
            else:
                roll  = np.arctan2(-R_SP[1, 2], R_SP[1, 1])
                pitch = np.arctan2(-R_SP[2, 0], sy)
                yaw   = 0.0


            roll, pitch, yaw = np.degrees([roll, pitch, yaw])
            
            # Draw tag outline
            corners = image_points.astype(int)
            for i in range(4):
                cv2.line(
                    frame,
                    tuple(corners[i]),
                    tuple(corners[(i + 1) % 4]),
                    (0, 255, 0),
                    2
                )

            # Draw coordinate axes
            axis_len = tagSize / 2.0
            axes_3d = np.float64([
                [axis_len, 0, 0],  # X
                [0, axis_len, 0],  # Y
                [0, 0, axis_len]   # Z
            ])
            rvec, _ = cv2.Rodrigues(R_SP)
            imgpts, _ = cv2.projectPoints(
                axes_3d, rvec, tvec, K, D
            )

            origin = tuple(corners.mean(axis=0).astype(int))

            cv2.line(frame, origin, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)   # X (red)
            cv2.line(frame, origin, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)   # Y (green)
            cv2.line(frame, origin, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)   # Z (blue)

            # Display pose info
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

        cv2.imshow("AprilTag Targeting (6-DOF)", frame)

        if cv2.waitKey(1) & 0xFF == 32:
            #save image
            break

    cap.release()
    cv2.destroyAllWindows()

    # Results/Storage
    results = []
    for tag_id in [0,1,2]:
        if tag_id in tag_measurements:
            results.append(tag_measurements[tag_id])
        else:
            results.append(np.full(6,np.nan))
    return np.array(results)

def CenterHatch(tagMeas):
    # Inputs: xyzrpy readings from all 3 tags from 1 camera
    # Outputs: Estimated center xyzrpy assuming the 3 tags form a circle
    
    # Unpacking data (Only need x, y, z)
    tag1, tag2, tag3 = tagMeas
    p1 = tag1[:3]
    p2 = tag2[:3]
    p3 = tag3[:3]
    # Calculating normal
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1,v2)

    # Perpendicular Bisectors
    n1 = np.cross(n,v1)
    n2 = np.cross(n,v2)

    # Midpoints
    m1 = (p1+p2) / 2
    m2 = (p1+p3) / 2

    # Center
    # m1 + alpha*N1 = m2 + beta*N2

    a = np.column_stack((n1,-n2))
    b = m2 - m1

    x,_,_,_ = np.linalg.lstsq(a,b,rcond=None)

    center = m1 + x[0]*n1

    return center

if __name__ == "__main__":
    # # AprilTag Measurements
    # tagMeas = CameraReadings()

    # Test Tag Values
    tag1 = np.array([-10,10,10,0,0,0])
    tag2 = np.array([10,10,10,0,0,0])
    tag3 = np.array([0,np.sqrt(200),10,0,0,0])
    tagMeas = np.stack((tag1, tag2, tag3))
    center = CenterHatch(tagMeas) # Array Shape (3, 6)
    print(center)

