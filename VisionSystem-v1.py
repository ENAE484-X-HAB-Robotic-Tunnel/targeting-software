import sys
import time
import math
import numpy as np
import RotationMatrix
from StewartPlatform import StewartPlatform
import cv2
from pupil_apriltags import Detector

"""
In this document the convention alpha_R_beta and alpha_t_beta is used to express R and t vectors to state alpha in frame beta
Notable states used are the tag and hatch state and frames used are the camera and measurement/platform frames. 
In general state changes occur in the tag class and frame changes occur in the platform class
See documentation for further details.

Inputs:
[Space] - Plot Hatch State
[S]     - Save Image (Disabled)
[ESC]   - Exit
"""


class Hatch:
    """
    Description: Passive docking target. Owns the geometric configuration of its
                 AprilTags and provides a lookup table for the platform to resolve
                 detections to their corresponding Tag objects.
    Attributes:
        - tags      : list of Tag objects mounted on this hatch
        - tag_lookup: dict mapping tag ID -> Tag, for O(1) resolution during detection
    Methods:
        - (none — purely a data container; pose estimation is handled by Platform)
    """
    def __init__(self, tags):
        self.tags = tags
        self.tag_lookup = {t.id: t for t in tags}

class Platform:
    """
    Description: Active chaser spacecraft. Owns one or more cameras and is responsible
                 for reading detections, transforming measurements into the platform
                 (docking) frame, fusing multi-tag and multi-camera estimates, and
                 driving the render loop. The platform measurement frame has x forward
                 (into hatch), y left, z up.
    Attributes:
        - cameras                : list of Camera objects mounted on the platform
        - target                 : the Hatch object being tracked
        - C_platformChangeOfBasis: 4x4 change-of-basis from camera convention to
                                   platform docking frame convention
    Methods:
        - updateTarget()         : runs one full measurement cycle across all cameras,
                                   returns fused (R, t) of target in platform frame
        - camToPlatformFrame()   : transforms a pose from a specific camera frame into
                                   the platform measurement frame
        - averagePoses()         : fuses a list of (R, t) pairs into a single estimate
                                   by averaging rvecs and translations
    """
    def __init__(self, cameras, target):
        self.cameras = cameras
        self.target = target

        self.C_platformChangeOfBasis = np.array([
            [ 0,  0,  1, 0],
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  0, 1]
        ])

    def updateTarget(self):
        poses_platform = []; target_R = None; target_t = None
        for cam in self.cameras:
            ret, frame, detections = cam.read()
            for det in detections:
                Renderer.drawTagOutline(frame, cam, det)
                Renderer.drawAxes(frame, cam, det.pose_R, det.pose_t)

            poses_cam = []
            for det in detections:
                if det.tag_id in self.target.tag_lookup:
                    tag = self.target.tag_lookup.get(det.tag_id)
                    target_R_cam, target_t_cam = tag.camToHatch(det.pose_R, det.pose_t)
                    poses_cam.append((target_R_cam, target_t_cam))

            if poses_cam:
                avg_R_cam, avg_t_cam = self.averagePoses(poses_cam)
                Renderer.drawAxes(frame, cam, avg_R_cam, avg_t_cam)
                avg_R_platform, avg_t_platform = self.camToPlatformFrame(cam, avg_R_cam, avg_t_cam)
                Renderer.draw_pose_panel(frame, cam, "Hatch (Docking Frame)", avg_R_platform, avg_t_platform, 50)
                poses_platform.append((avg_R_platform, avg_t_platform))

            Renderer.renderFrame(frame)

        if poses_platform:
            target_R, target_t = self.averagePoses(poses_platform)
        return target_R, target_t

    def camToPlatformFrame(self, cam, pose_R_cam, pose_t_cam):
        T_camToHatch = Transform.from_R_t(pose_R_cam, pose_t_cam)
        T_platformToHatch = self.C_platformChangeOfBasis @ cam.T_platformToCam @ T_camToHatch
        return Transform.to_R_t(T_platformToHatch)

    def averagePoses(self, poses):
        if not poses: return None
        rotations    = [pose[0] for pose in poses]
        translations = [pose[1].reshape(3) for pose in poses]

        rvecs    = [cv2.Rodrigues(R)[0] for R in rotations]
        avg_rvec = np.mean(rvecs, axis=0)
        avg_R, _ = cv2.Rodrigues(avg_rvec)
        avg_t    = np.mean(translations, axis=0)

        return avg_R, avg_t

class Tag:
    """
    Description: A single AprilTag mounted on the hatch rim at a known radial distance
                 and angular position. Owns the precomputed transform chain that maps a
                 raw camera detection of this tag into the hatch-centre frame.
                 Tags are assumed installed with +y pointing radially outward; tagAngle
                 is measured clockwise from vertical.
    Attributes:
        - id                  : int, AprilTag ID
        - tagSize             : class-level float, physical side length of all tags in metres
        - T_tagToHatch        : 4x4 homogeneous transform — positions this tag's origin
                                relative to the hatch centre (translation + angular alignment)
        - C_hatchChangeOfBasis: 4x4 change-of-basis from tag-camera convention to
                                hatch frame convention
    Methods:
        - camToHatch()        : given a raw detector pose (R, t) in camera frame, returns
                                the hatch-centre pose (R, t) in camera frame
    """
    tagSize = 0.05

    def __init__(self, id, tagRad, tagAngle):
        self.id = id

        T_center = Transform.from_R_t(np.eye(3), np.array([0, tagRad, 0]))
        T_align  = Transform.from_R_t(RotationMatrix.rotZ(-tagAngle), np.array([0, 0, 0]))
        self.C_hatchChangeOfBasis = np.array([
            [ 0,  0,  1, 0],
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  0, 1]
        ]).T

        self.T_tagToHatch = T_center @ T_align

    def camToHatch(self, pose_R, pose_t):
        T_camToTag   = Transform.from_R_t(pose_R, pose_t)
        T_camToHatch = T_camToTag @ self.T_tagToHatch @ self.C_hatchChangeOfBasis
        return Transform.to_R_t(T_camToHatch)

class Camera:
    """
    Description: A single camera mounted at a known position and orientation on the
                 platform. Owns its intrinsics, distortion coefficients, the video
                 capture handle, and the AprilTag detector. Provides a read() method
                 that returns a frame and its detections. The extrinsic T_platformToCam
                 encodes where this camera sits relative to the platform measurement
                 frame, allowing the Platform to transform each camera's measurements
                 into a common frame before fusion.
    Attributes:
        - id             : int, OpenCV camera index
        - fx, fy, cx, cy : intrinsic focal lengths and principal point (pixels)
        - cameraMatrix   : 3x3 numpy intrinsic matrix
        - distortion     : (1,5) distortion coefficients [k1, k2, p1, p2, k3]
        - T_platformToCam: 4x4 extrinsic — platform frame expressed in camera frame
        - cap            : cv2.VideoCapture handle
        - detector       : pupil_apriltags Detector instance
    Methods:
        - read()         : captures one frame, runs detection, returns (ret, frame, detections)
        - exit()         : releases the video capture handle
    """
    def __init__(self, id, cameraParams, cameraDist, cam_R, cam_t):
        self.id = id
        self.fx, self.fy, self.cx, self.cy = cameraParams
        self.cameraMatrix = np.array([[self.fx, 0, self.cx],[0, self.fy, self.cy],[0, 0, 1]], dtype=np.float64)
        self.distortion   = cameraDist

        self.T_platformToCam = Transform.from_R_t(cam_R, cam_t)

        self.cap = cv2.VideoCapture(id)
        if not self.cap.isOpened():
            sys.exit(f"[ERROR] Cannot open camera index {id}")

        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1,
                                 quad_sigma=0.0, refine_edges=True, decode_sharpening=0.25)

    def read(self):
        ret, frame = self.cap.read()
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray, estimate_tag_pose=True,
                                          camera_params=(self.fx, self.fy, self.cx, self.cy),
                                          tag_size=Tag.tagSize)
        return ret, frame, detections

    def exit(self):
        self.cap.release()

class Renderer:
    @staticmethod
    def renderFrame(frame):
        cv2.imshow("Hatch 6DOF Targetter", frame)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s'):
        #     fname = f"hatch_capture_{int(time.time())}.png"
        #     cv2.imwrite(fname, frame)
        #     print(f"Saved {fname}")

    @staticmethod
    def drawTagOutline(frame, cam, det):
        pts = det.corners.astype(int)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,220,255), 2, cv2.LINE_AA)
        cx_ = int(pts[:,0].mean()); cy_ = int(pts[:,1].mean())
        label = f"ID:{det.tag_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (cx_-tw//2-4, cy_-th-4), (cx_+tw//2+4, cy_+4), (0,0,0), -1)
        cv2.putText(frame, label, (cx_-tw//2, cy_), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,255), 2, cv2.LINE_AA)

    @staticmethod
    def drawAxes(frame, cam, R, tvec):
        axis_len = 0.05
        axis_3d  = np.float32([[axis_len,0,0],[0,axis_len,0],[0,0,axis_len],[0,0,0]])
        rvec, _  = cv2.Rodrigues(R)
        pts, _   = cv2.projectPoints(axis_3d, rvec, tvec, cam.cameraMatrix, cam.distortion)
        pts      = pts.reshape(-1, 2).astype(int)
        origin   = tuple(pts[3])
        cv2.arrowedLine(frame, origin, tuple(pts[0]), (0,  0,  255), 3, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(pts[1]), (0,  255,  0), 3, cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(frame, origin, tuple(pts[2]), (255,  0,  0), 3, cv2.LINE_AA, tipLength=0.3)
        for label, pt, color in [("X",pts[0],(0,0,255)),("Y",pts[1],(0,255,0)),("Z",pts[2],(255,0,0))]:
            cv2.putText(frame, label, (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_pose_panel(frame, cam, objName, R, tvec, panel_y):
        tx, ty, tz       = tvec.flatten()
        roll, pitch, yaw = [180/math.pi*i for i in RotationMatrix.R_to_euler(R)]
        lines = [f"{objName}  (docking frame)",
                 f" x={tx:+.3f}m  y={ty:+.3f}m  z={tz:+.3f}m   roll={roll:+.1f}  pitch={pitch:+.1f}  yaw={yaw:+.1f} deg"]
        h, w  = frame.shape[:2]
        y0    = panel_y - len(lines)*22 - 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,y0), (w,panel_y), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10,y0+20+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230,230,230), 1, cv2.LINE_AA)
        return y0

class Transform:
    @staticmethod
    def from_R_t(R, t):
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t.flatten()
        return T

    @staticmethod
    def to_R_t(T):
        return T[:3,:3], T[:3,3]

def main():
    fx, fy = 800, 800; cx, cy = 300, 200
    cam_R = np.eye(3); cam_t = np.array([0, 0, 0])
    dist_coeffs = np.array([[0.07863, -0.20173, -0.00186, 0.00294, 0.21325]], dtype=np.float64)
    cam0 = Camera(0, [fx, fy, cx, cy], dist_coeffs, cam_R, cam_t)
    cams = [cam0]

    tag0 = Tag(0, 0.13, 1*math.pi/3)
    tag1 = Tag(1, 0.13, 3*math.pi/3)
    tag2 = Tag(2, 0.13, 5*math.pi/3)
    tags = [tag0, tag1, tag2]

    hatch    = Hatch(tags)
    platform = Platform(cams, hatch)

    while True:
        target_R, target_t = platform.updateTarget()

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break  # ESC to close
        elif key == 32:
            if target_R is not None and target_t is not None:
                x, y, z = target_t
                roll, pitch, yaw = RotationMatrix.R_to_euler(target_R)
                SP = StewartPlatform(np.array([x, y, z, roll, pitch, yaw]), 0.5, 0.5)
                SP.plot()
                break

    for cam in cams: cam.exit()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()