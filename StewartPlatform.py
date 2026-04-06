import numpy as np
import math
import RotationMatrix
import matplotlib.pyplot as plt

class StewartPlatform:
    def __init__(self, X, rad_base, rad_platform):
        x, y, z, roll, pitch, yaw = X
        self.rad_base = rad_base; self.rad_platform = rad_platform
        self.pose_R = RotationMatrix.euler_to_R(roll, pitch, yaw)
        self.pose_t = X[:3]
        self.T = np.eye(4); self.T[:3, :3] = self.pose_R; self.T[:3, 3] = self.pose_t.T
        
        self.B, self.P = self.jointLocations(rad_base, rad_platform)
        self.P_B, self.l_legs = self.inverseKinematics(self.B, self.P, self.T) # Platform joint locations in Base frame

    def inverseKinematics(self, B, P, T):
        """
        B: (3,6) base joints
        P: (3,6) platform joints (platform frame)
        T: (4,4) transform from platform → base
        """

        # Convert P to homogeneous coordinates (4x6)
        P_h = np.vstack((P, np.ones((1, 6))))

        # Transform all platform joints into base frame
        P_B_h = T @ P_h

        # Drop homogeneous row → (3x6)
        P_B = P_B_h[:3, :]

        # Compute leg lengths
        l_legs = np.linalg.norm(P_B - B, axis=0)

        return P_B, l_legs

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ── Legs ──
        for i in range(6):
            ax.plot(
                [self.B[0,i], self.P_B[0,i]],
                [self.B[1,i], self.P_B[1,i]],
                [self.B[2,i], self.P_B[2,i]],
                color='blue', linewidth=2
            )

        # ── Base hexagon (connect joints in order, close the loop) ──
        Bx = list(self.B[0,:]) + [self.B[0,0]]
        By = list(self.B[1,:]) + [self.B[1,0]]
        Bz = list(self.B[2,:]) + [self.B[2,0]]
        ax.plot(Bx, By, Bz, color='black', label='Base')

        # ── Platform hexagon (P_B: platform joints in base frame) ──
        Px = list(self.P_B[0,:]) + [self.P_B[0,0]]
        Py = list(self.P_B[1,:]) + [self.P_B[1,0]]
        Pz = list(self.P_B[2,:]) + [self.P_B[2,0]]
        ax.plot(Px, Py, Pz, color='magenta', label='Platform')

        # ── Base circle ──
        theta = np.linspace(0, 2*np.pi, 100)
        # Base lies in YZ plane (x-axis is extension direction), so circle is in local XY
        # but base frame has no rotation — just draw in XY at z=0
        bcirc = np.array([
            np.zeros_like(theta),
            self.rad_base * np.cos(theta),
            self.rad_base * np.sin(theta),
        ])
        ax.plot(bcirc[0], bcirc[1], bcirc[2], color='black', linestyle='--')

        # ── Platform circle (transform unit circle into base frame via self.T) ──
        pcirc_local = np.vstack([
            np.zeros_like(theta),
            self.rad_platform * np.cos(theta),
            self.rad_platform * np.sin(theta),
            np.ones_like(theta)
        ])
        pcirc_base = (self.T @ pcirc_local)[:3, :]
        ax.plot(pcirc_base[0], pcirc_base[1], pcirc_base[2], color='magenta', linestyle='--')

        # ── Platform normal vector ──
        normal_base = self.pose_R @ np.array([1, 0, 0])
        ax.quiver(
            self.pose_t[0], self.pose_t[1], self.pose_t[2],
            normal_base[0], normal_base[1], normal_base[2],
            length=0.2, color='green', linewidth=2,
            arrow_length_ratio=0.1, label='Normal'
        )

        # ── Tunnel cylinder ──
        vis_r = self.rad_base * 0.8
        base_circ   = np.vstack([np.zeros_like(theta), vis_r * np.cos(theta), vis_r * np.sin(theta)])
        plat_circ_h = np.vstack([np.zeros_like(theta), vis_r * np.cos(theta), vis_r * np.sin(theta), np.ones_like(theta)])
        plat_circ   = (self.T @ plat_circ_h)[:3, :]               # platform circle in base frame

        X_surf = np.vstack([base_circ[0], plat_circ[0]])
        Y_surf = np.vstack([base_circ[1], plat_circ[1]])
        Z_surf = np.vstack([base_circ[2], plat_circ[2]])
        ax.plot_surface(X_surf, Y_surf, Z_surf, color='blue', alpha=0.3)

        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.axis('equal')
        plt.show()

    def jointLocations(self, rad_base, rad_platform):
        # Returns base and platform joint locations in their respective frames
        B = np.zeros((3, 6))
        P = np.zeros((3, 6))

        jointSpacing = 0  # Joint pair spacing

        alpha_base = math.asin((jointSpacing / 2) / rad_base) if rad_base != 0 else 0
        alpha_platform = math.asin((jointSpacing / 2) / rad_platform) if rad_platform != 0 else 0
        
        # --- Base joints ---
        B[:, 0] = [0, rad_base * np.cos((1*np.pi/6) - alpha_base), rad_base * np.sin((1*np.pi/6) - alpha_base)]
        B[:, 1] = [0, rad_base * np.cos((1*np.pi/6) + alpha_base), rad_base * np.sin((1*np.pi/6) + alpha_base)]
        B[:, 2] = [0, rad_base * np.cos((5*np.pi/6) - alpha_base), rad_base * np.sin((5*np.pi/6) - alpha_base)]
        B[:, 3] = [0, rad_base * np.cos((5*np.pi/6) + alpha_base), rad_base * np.sin((5*np.pi/6) + alpha_base)]
        B[:, 4] = [0, rad_base * np.cos((9*np.pi/6) - alpha_base), rad_base * np.sin((9*np.pi/6) - alpha_base)]
        B[:, 5] = [0, rad_base * np.cos((9*np.pi/6) + alpha_base), rad_base * np.sin((9*np.pi/6) + alpha_base)]

        # --- Platform joints ---
        P[:, 0] = [0, rad_platform * np.cos((11*np.pi/6) + alpha_platform), rad_platform * np.sin((11*np.pi/6) + alpha_platform)]
        P[:, 1] = [0, rad_platform * np.cos((3*np.pi/6)  - alpha_platform), rad_platform * np.sin((3*np.pi/6)  - alpha_platform)]
        P[:, 2] = [0, rad_platform * np.cos((3*np.pi/6)  + alpha_platform), rad_platform * np.sin((3*np.pi/6)  + alpha_platform)]
        P[:, 3] = [0, rad_platform * np.cos((7*np.pi/6)  - alpha_platform), rad_platform * np.sin((7*np.pi/6)  - alpha_platform)]
        P[:, 4] = [0, rad_platform * np.cos((7*np.pi/6)  + alpha_platform), rad_platform * np.sin((7*np.pi/6)  + alpha_platform)]
        P[:, 5] = [0, rad_platform * np.cos((11*np.pi/6) - alpha_platform), rad_platform * np.sin((11*np.pi/6) - alpha_platform)]
        return B, P
    

if __name__ == '__main__':
    X = np.array([2,0,0,0,0,0])
    SP = StewartPlatform(X, 1, 1)
    SP.plot()