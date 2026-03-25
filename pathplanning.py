import pupil_apriltags
import cv2
import numpy as np
import time
import traceback
from queue import Empty
from robomaster import robot
from robomaster import camera

import dijkstras as path

MAP = "map.csv"

"""
 	-> x
 |
 v  y

 z is clockwise rotation from x
 
"""

# first define tag information and then obtain their transforms
tag_info = {
	# format: tag_id: (x * scale, y * scale, theta_degrees)
	34: (2 * 26.6, 4 * 26.6, 90),
	33: (2 * 26.6, 5 * 26.6, 0),
	32: (2 * 26.6, 5 * 26.6, 180),
	31: (2 * 26.6, 7 * 26.6, 0),
	30: (2 * 26.6, 7 * 26.6, 180),
	35: (4 * 26.6, 8 * 26.6, 90),
	41: (5 * 26.6, 1 * 26.6, 0),
	40: (5 * 26.6, 1 * 26.6, 180),
	39: (5 * 26.6, 3 * 26.6, 0),
	38: (5 * 26.6, 3 * 26.6, 180),
	37: (5 * 26.6, 4 * 26.6, 270),
	36: (6 * 26.6, 8 * 26.6, 90),
	46: (8 * 26.6, 4 * 26.6, 90),
	45: (8 * 26.6, 5 * 26.6, 0),
	44: (8 * 26.6, 5 * 26.6, 180),
	43: (8 * 26.6, 7 * 26.6, 0),
	42: (8 * 26.6, 7 * 26.6, 180)
}

def create_2d_transform(x_m, y_m, theta_deg):
	"""
	input:
	x: meters
	y: meters
	theta: degrees

	output:
	4x4 transformation matrix for a 2d pose
	"""

	theta_rad = np.radians(theta_deg)
	T = np.eye(4)
	T[0, 0] = np.cos(theta_rad)
	T[0, 1] = -np.sin(theta_rad)
	T[1, 0] = np.sin(theta_rad)
	T[0, 1] = np.cos(theta_rad)
	T[0, 3] = x_m
	T[1, 3] = y_m
	return T
	 
# transform each tag to base frame
scale_m = 0.266

tag_transforms = {
	tag_id: create_2d_transform(x_scale * scale_m, y_scale * scale_m, theta)
	for tag_id, (x_scale, y_scale, theta) in tag_info.items()
}

def get_base_robot_pose(tag_id, R_ca, t_ca, tag_transformations):
	if tag_id not in tag_transformations:
		# just incase we somehow fail to detect a valid april tag
		print('detected invalid tag')
		return None, None, None
	
	# find the tag pose in camera frame: T_C^tag
	T_C_tag = np.eye(4)
	T_C_tag[:3, :3] = R_ca
	T_C_tag[3:, 3:] = t_ca

	# find tage pose in base frame
	T_B_tag = tag_transforms[tag_id]

	# find camera (robot) pose in base frame T_B^C = T_B^tag T_C^tag, T_C^tag = inv(T_tag^C)
	T_B_C = T_B_tag @ np.linalg.inv(T_C_tag)

	# obtain robot pose in base frame
	base_x = T_B_C[0, 3]
	base_y = T_B_C[1, 3]
	base_yaw = np.atan2(T_B_C[1, 0], T_B_C[0, 0])

	return base_x, base_y, base_yaw


class AprilTagDetector:
	def __init__(self, K, family="tag36h11", threads=2, marker_size_m=0.16):
		self.camera_params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
		self.marker_size_m = marker_size_m
		self.detector = pupil_apriltags.Detector(family, threads)

	def find_tags(self, frame_gray):
		detections = self.detector.detect(frame_gray, estimate_tag_pose=True,
			camera_params=self.camera_params, tag_size=self.marker_size_m)
		return detections

def get_pose_apriltag_in_camera_frame(detection):
	R_ca = detection.pose_R
	t_ca = detection.pose_t
	return t_ca.flatten(), R_ca

def draw_detections(frame, detections):
	for detection in detections:
		pts = detection.corners.reshape((-1, 1, 2)).astype(np.int32)

		frame = cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

		top_left = tuple(pts[0][0])  # First corner
		top_right = tuple(pts[1][0])  # Second corner
		bottom_right = tuple(pts[2][0])  # Third corner
		bottom_left = tuple(pts[3][0])  # Fourth corner
		cv2.line(frame, top_left, bottom_right, color=(0, 0, 255), thickness=2)
		cv2.line(frame, top_right, bottom_left, color=(0, 0, 255), thickness=2)

def detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag):

	# obtain dijkstra's path
	target_path = path.main()

	# tolerance to the center of each node
	tolerance_m = 0.05 # 5 cm

	while True:
		try:
			img = ep_camera.read_cv2_image(strategy="newest", timeout=0.5)
		except Empty:
			time.sleep(0.001)
			continue

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray.astype(np.uint8)

		detections = apriltag.find_tags(gray)

		if len(detections) > 0:
			# no longer aborting when detecting more than one tag, always track
			# the first april tag detected
			detection = detections[0]
			tag_id = detection.tag_id

			t_ca, R_ca = get_pose_apriltag_in_camera_frame(detection)
			print('t_ca', t_ca)
			robot_x, robot_y, robot_yaw = get_base_robot_pose(tag_id, R_ca, t_ca, tag_transforms)
	
			# make sure that a valid pose was obtained
			if robot_x is not None:
				target_x, target_y = target_path[0]

				# check if we reached waypoint:
				# we are allowing the robot to be tolernace_m away from the target
				# node, so essentially drawing out a circle
				dist_to_target = np.norm(target_x - robot_x, target_y - robot_y)
				if dist_to_target < tolerance_m:
					# if the distance is less than the tolerance, we have reached
					# the current target node and can work on the next node

					# pop current node and set target to next node
					target_path.pop(0)

					# check to see if we reach end of path
					if len(target_path) == 0:
						print('goal reached!!!')
						ep_chassis.drive_speed(x=0, y=0, z=0)
						continue
					# else: set next node to target
					target_x, target_y = target_path[0]

			# translational control

			# obtain error in base frame
			error_x = target_x - robot_x
			error_y = target_y - robot_y

			# translate error to robot frame (Rz)
			rob_x =  error_x * np.cos(robot_yaw) + error_y * np.sin(robot_yaw)
			rob_y = -error_x * np.sin(robot_yaw) + error_y * np.cos(robot_yaw)

			# compute requried velocity based off gains
			# limit velocity to -1, 1
			# gains
			KP_TRANS = 1
			KP_ROT = 2
			
			x_vel = np.clip(KP_TRANS * rob_x, -1.0, 1.0)
			y_vel = np.clip(KP_TRANS * rob_y, -1.0, 1.0)


			# rotational control
			# we want to keep a tag in view at all times
			tag_base_x = tag_transforms[tag_id][0, 3]
			tag_base_y = tag_transforms[tag_id][1, 3]

			target_yaw = np.atan2(tag_base_y - robot_y, tag_base_x - robot_x)
			yaw_error = target_yaw - robot_yaw
			yaw_error = np.atan2(np.sin(yaw_error), np.cos(yaw_error))

			yaw_vel = np.clip(KP_ROT * np.rad2deg(yaw_error), -30.0, 30.0)

			# pass output
			ep_chassis.drive_speed(x=x_vel, y=y_vel, z = yaw_vel, timeout = 0.1)

			# check output
			print(f"Pose: X:{robot_x:.2f} Y:{robot_y:.2f} |  target pose: X:{target_x:.2f} Y:{target_y:.2f} | velocity control: X:{x_vel:.2f} Y:{y_vel:.2f} Z:{yaw_vel:.2f}")



		elif len(detections) == 0: 
			# if there are no tags rotate until we see a tag
			ep_chassis.drive_speed(x=0, y=0, z=1.0, timeout = 0.1)

		draw_detections(img, detections)
		cv2.imshow("img", img)
		if cv2.waitKey(1) == ord('q'):
			break



if __name__ == '__main__':
	# More legible printing from numpy.
	np.set_printoptions(precision=3, suppress=True, linewidth=120)

	ep_robot = robot.Robot()
	ep_robot.initialize(conn_type="ap")#(conn_type="sta", sn="3JKCH7T00100J0")
	ep_chassis = ep_robot.chassis
	ep_camera = ep_robot.camera
	ep_camera.start_video_stream(display=False, resolution=camera.STREAM_360P)

	K = np.array([[314, 0, 320], [0, 314, 180], [0, 0, 1]]) # Camera focal length and center pixel
	marker_size_m = 0.153 # Size of the AprilTag in meters
	apriltag = AprilTagDetector(K, threads=2, marker_size_m=marker_size_m)

	try:
		detect_tag_loop(ep_robot, ep_chassis, ep_camera, apriltag)
	except KeyboardInterrupt:
		pass
	except Exception as e:
		print(traceback.format_exc())
	finally:
		print('Waiting for robomaster shutdown')
		ep_camera.stop_video_stream()
		ep_robot.close()
