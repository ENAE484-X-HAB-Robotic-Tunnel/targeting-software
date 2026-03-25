import cv2
import numpy as np
import matplotlib.pyplot as plt
import ik_visualization
import os
from pupil_apriltags import Detector



def aprilTagTargetting():
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
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # filename = os.path.join(script_dir, "TargetAprilImage.jpg")
    # cv2.imwrite(filename, frame)
    cv2.destroyAllWindows()
    targetState = np.array([x, y, z, roll, pitch, yaw])
    return targetState

def plotSP(x, y, z, roll, pitch, yaw):
    base_r = 5; platform_r = 5
    platform = ik_visualization.StewartPlatform33(base_r, platform_r) # give base and platform radius
    ik_visualization.platform = platform

    # x y z, r p y in deg
    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]

    target_pos = [x, y, z] # extension, translation in yz plane
    target_rpy = [roll, pitch + 90, yaw] # deg
    ik_visualization.target_pos = target_pos
    ik_visualization.target_rpy = target_rpy

    lengths, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, base_rpy, target_pos, target_rpy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ik_visualization.ax = ax

    # plot legs
    # manage connectivity, start and end position of each line in x, y, and z
    for name, (start, end) in lines.items():
        ax.plot(
            [start[0], end[0]], # x
            [start[1], end[1]], # y
            [start[2], end[2]], # z
            color='blue', linewidth=2
        )

    # plot triangle connecting legs on each platform
    platform.plot_triangle(base_pts, ['A', 'B', 'C'], 'black', 'Base')
    platform.plot_triangle(plat_pts, ['D', 'E', 'F'], 'magenta', 'Platform')

    # plot platforms
    platform.plot_circle(base_pos, platform.r_base, base_rpy, 'black')
    platform.plot_circle(target_pos, platform.r_plat, target_rpy, 'magenta')

    # plot the Normal Vector
    platform.plot_normal(ax, target_rpy)

    
    # plot cylinder to visualize tunnel
    vis_r = base_r * 0.8
    platform.plot_cylinder(ax, platform, base_pos, base_rpy, target_pos, target_rpy, vis_r)

    
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.axis('equal')
    plt.show()


if __name__ == "__main__":
    pose = aprilTagTargetting()
    multiplier = 10
    
    if pose is not None:
        x, y, z, roll, pitch, yaw = pose
        x, y, z = x*multiplier, y*multiplier, z*multiplier
        print("\nCaptured Pose:")
        print(f"x={x:.3f} y={y:.3f} z={z:.3f}")
        print(f"roll={roll:.2f} pitch={pitch:.2f} yaw={yaw:.2f}")
        plotSP(x, y, z, roll, pitch, yaw)
    else:
        print("No pose captured.")