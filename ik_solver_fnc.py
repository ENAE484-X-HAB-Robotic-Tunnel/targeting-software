"""
Derivation from Journal of Mechatronics and Robotics, Inverse Kinematics of a Stewart Platform by R. Petrescu, et al. Oct 2018
and 
The Mathematics of the Stewart Platform by Radamés Ajna and Thiago Hersan as part of memememe for the Object Liberation Front https://olf.alab.space/projects/


Solves for the leg lengths required for the Stewart Platform to reach a given x y z roll pitch yaw
"""

import numpy as np
import matplotlib.pyplot as plt

class StewartPlatform33:
    def __init__(self, base_r, plat_r):
        """
        Initialize the geometry of a 3-3 Stewart Platform
        """

        self.base_r = base_r
        self.plat_r = plat_r

        # Defining local base points, relative to the Base Center
        # Equilaterial Triangle
        # A is at 0 degrees, B is at 120, C is at 240.
        self.local_base_points = {
            'A': np.array([base_r, 0, 0]),
            'B': np.array([-0.5 * base_r, np.sqrt(3)/2 * base_r, 0]),  # Fixed: +Y for 120 deg
            'C': np.array([-0.5 * base_r, -np.sqrt(3)/2 * base_r, 0])  # Fixed: -Y for 240 deg
        }

        # Defining local platform points, relative to platform center
        # D goes to A and B, E goes to B and C, F goes to C and A

        # Top platform is rotated by 60 degrees for 3-3 geometry
        angle = np.deg2rad(60)

        Rz_60 = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])

        self.local_plat_points = {
            'D': Rz_60 @ np.array([plat_r, 0, 0]),
            'E': Rz_60 @ np.array([-0.5 * plat_r, np.sqrt(3)/2 * plat_r, 0]),
            'F': Rz_60 @ np.array([-0.5 * plat_r, -np.sqrt(3)/2 * plat_r, 0])
        }

    def get_rpy(self, rpy):
        # obtain rotation matrix given row pitch and yaw
        rpy = np.deg2rad(rpy)
        roll, pitch, yaw = rpy
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx
    
    def solve_leg_lengths(self, base_pos, base_rpy, target_pos, target_rpy):
        # 321 rotation matrix from base to target frame
        base_r = self.get_rpy(base_rpy)
        plat_r = self.get_rpy(target_rpy)

        # define how the legs are connected
        leg_con = [
            ('Leg1', 'A', 'D'), # Leg 1 connects base point A to plat point D
            ('Leg2', 'B', 'D'), 
            ('Leg3', 'B', 'E'), 
            ('Leg4', 'C', 'E'),
            ('Leg5', 'C', 'F'),
            ('Leg6', 'A', 'F')
        ]

        # ik solution, leg length and leg vector
        lengths = {}
        lines = {}
        
        # storing coordinates to plot platform triangles
        base_coords = {}
        plat_coords = {}

        # target position and base position in base frame
        T = np.array(target_pos) - np.array(base_pos)
        B = np.array(base_pos)

        # solve for leg lengths
        for leg_name, base_key, plat_key in leg_con:
            # start by obtaining local coordinates p_i and b_i
            b_i = self.local_base_points[base_key]
            p_i = self.local_plat_points[plat_key]

            # Convert to target frame to base frame
            base_point = B + (base_r @ b_i)
            plat_point = T + (plat_r @ p_i)
            
            # Store point coords for plotting
            base_coords[base_key] = base_point
            plat_coords[plat_key] = plat_point

            # find leg lengths
            leg_vec = plat_point - base_point

            lengths[leg_name] = np.linalg.norm(leg_vec)
            lines[leg_name] = (base_point, plat_point)

        return lengths, lines, base_coords, plat_coords
    
    def plot_triangle(self, ax, points, keys, color, label):
        # draw a triangle from going to a -> b -> c -> a
        x = [points[k][0] for k in keys] + [points[keys[0]][0]] # A_x -> B_x -> C_x -> A_x
        y = [points[k][1] for k in keys] + [points[keys[0]][1]] # repeat for y
        z = [points[k][2] for k in keys] + [points[keys[0]][2]] # repeat for z
        ax.plot(x, y, z, color=color, label=label)

    def plot_circle(self, ax, center, r, rpy=[0,0,0], color='black'):
        # draw platform circle
        theta = np.linspace(0, 2*np.pi, 100)
        # Create circle in local XY plane
        xc = r * np.cos(theta) # x coord of circle
        yc = r * np.sin(theta) # y
        zc = np.zeros_like(xc) # z
        
        # apply rotation and translation
        points = np.vstack([xc, yc, zc])
        R = self.get_rpy(rpy) # get rotation matrix
        points = R @ points + np.array(center).reshape(3, 1)
        
        ax.plot(points[0,:], points[1,:], points[2,:], color=color, linestyle='--')

    def plot_normal(self, ax, target_pos, target_rpy):
        # plot normal vector of target
        plat_r = self.get_rpy(target_rpy)
        platform_normal = np.array([0, 0, 1]) # define normal of platform

        # transform normal from platform frame to base frame
        base_frame_normal = plat_r @ platform_normal
        
        ax.quiver(
                target_pos[0], target_pos[1], target_pos[2],
                base_frame_normal[0], base_frame_normal[1], base_frame_normal[2],
                length=1,
                color='green', 
                linewidth=2,
                arrow_length_ratio=0.1,
                label='Normal Vector'
            )

    def plot_cylinder(self, ax, base_pos, base_rpy, target_pos, target_rpy, radius, color='blue'):
        # plotting the usable tunnel


        # create base of cylinder
        theta = np.linspace(0, 2*np.pi, 50)
        xb = radius * np.cos(theta)
        yb = radius * np.sin(theta)
        zb = np.zeros_like(xb)
        base_points = np.vstack([xb, yb, zb])

        # rotate and translate to match base orientation
        base_r = self.get_rpy(base_rpy)
        base_points = (base_r @ base_points) + np.array(base_pos).reshape(3, 1)

        xt = radius * np.cos(theta)
        yt = radius * np.sin(theta)
        zt = np.zeros_like(xt)
        target_points = np.vstack([xt, yt, zt])

        # rotate and translate from base frame to target frame
        R_target = self.get_rpy(target_rpy)
        target_points = (R_target @ target_points) + np.array(target_pos).reshape(3, 1)

        # create a surface mesh to connect all the points

        X = np.vstack([base_points[0], target_points[0]])
        Y = np.vstack([base_points[1], target_points[1]])
        Z = np.vstack([base_points[2], target_points[2]])

        ax.plot_surface(X, Y, Z, color = color, alpha=0.3)


def Stewart_Solver(target_pos,target_rpy):
    base_r = 25
    platform_r = 25

    platform = StewartPlatform33(base_r, platform_r) # give base and platform radius

    # give the target coords and rpy in deg
    # x y z
    # platform extends along the x axis, and is rotated 90 degrees about the y axis to become a tunnel instead of a tower
    base_pos = [0, 0, 0]
    base_rpy = [0, 90, 0]

    # adjust as needed
    #target_pos = [35, 15, 15] # extension, translation in yz plane
    #target_rpy = [30, 90, 0] # deg

    lengths, lines, base_pts, plat_pts = platform.solve_leg_lengths(base_pos, base_rpy, target_pos, target_rpy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


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
    platform.plot_triangle(ax, base_pts, ['A', 'B', 'C'], 'black', 'Base')
    platform.plot_triangle(ax, plat_pts, ['D', 'E', 'F'], 'magenta', 'Platform')

    # plot platforms
    platform.plot_circle(ax, base_pos, platform.base_r, base_rpy, 'black')
    platform.plot_circle(ax, target_pos, platform.plat_r, target_rpy, 'magenta')

    # plot the Normal Vector
    platform.plot_normal(ax, target_pos, target_rpy)

    
    # plot cylinder to visualize tunnel
    vis_r = base_r * 0.8
    platform.plot_cylinder(ax, base_pos, base_rpy, target_pos, target_rpy, vis_r)

    
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.axis('equal')
    plt.show()

