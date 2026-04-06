import math
import numpy as np


def R_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll  = math.atan2( R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2( R[1, 0], R[0, 0])
    else:
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw

def euler_to_R(phi, theta, psi):
    return rotZ(psi)@rotY(theta)@rotX(phi)

def rotX(phi):
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rotY(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rotZ(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

# class RotationMatrix:
#     @staticmethod 
#     def R_to_euler(R):
#         sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
#         if sy > 1e-6:
#             roll  = math.atan2( R[2, 1], R[2, 2])
#             pitch = math.atan2(-R[2, 0], sy)
#             yaw   = math.atan2( R[1, 0], R[0, 0])
#         else:
#             roll  = math.atan2(-R[1, 2], R[1, 1])
#             pitch = math.atan2(-R[2, 0], sy)
#             yaw   = 0.0
#         return roll, pitch, yaw
    
#     @staticmethod 
#     def euler_to_R(phi, theta, psi):
#         return RotationMatrix.rotZ(psi)@RotationMatrix.rotY(theta)@RotationMatrix.rotX(phi)

#     @staticmethod 
#     def rotX(phi):
#         c, s = math.cos(phi), math.sin(phi)
#         return np.array([[1,0,0],[0,c,-s],[0,s,c]])

#     @staticmethod 
#     def rotY(theta):
#         c, s = math.cos(theta), math.sin(theta)
#         return np.array([[c,0,s],[0,1,0],[-s,0,c]])

#     @staticmethod 
#     def rotZ(psi):
#         c, s = math.cos(psi), math.sin(psi)
#         return np.array([[c,-s,0],[s,c,0],[0,0,1]])