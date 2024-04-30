import numpy as np


class Scorbot:
    num_joints = 4
    save_space = [[-130, 130], [-70, 70], [], []]  # angel limits of base, shoulder, elbow, and wrist in degree

    # Home parameters
    home_q1 = 23.6 * (np.pi / 180)
    home_q2 = -22.0 * (np.pi / 180)
    home_q3 = 22.4 * (np.pi / 180)

    def __init__(self):
        pass

    def direct_kinematics(self, q1=0, q2=0, q3=0, q4=0):
        # Here x, y and z are the Cartesian coordinates of the tip of the end-effector, and
        # q1 = angle of the base joint,
        # q2 = angle of the shoulder joint,
        # q3 = angle of the elbow joint,
        # q4 = angle of the wrist joint.
        # All angles are in radians.
        # To get q1, q2, q3, q4 from the angular velocities, I use: np.cumsum(angular_velocities/1000, axis=0).

        d1 = 0.3585
        a2 = 0.300
        a3 = 0.350
        a4 = 0.22
        d2 = -0.098
        a1 = 0.050
        d3 = 0.065

        q1 = -(q1 + self.home_q1)
        q2 = -(q2 + self.home_q2)
        q3 = -(q3 + self.home_q3)

        x = a1 * np.cos(q1 + 3745578344 / 9093467413) + a2 * np.cos(q1 + 3745578344 / 9093467413) * np.cos(
            q2 - 3357166075 / 8743247601) - a3 * np.sin(q2 - 3357166075 / 8743247601) * np.sin(
            q3 + 1421104251 / 3634967671) * np.cos(q1 + 3745578344 / 9093467413) + a3 * np.cos(
            q1 + 3745578344 / 9093467413) * np.cos(q2 - 3357166075 / 8743247601) * np.cos(
            q3 + 1421104251 / 3634967671) + a4 * (
                -np.sin(q2 - 3357166075 / 8743247601) * np.sin(q3 + 1421104251 / 3634967671) * np.cos(
                    q1 + 3745578344 / 9093467413) + np.cos(q1 + 3745578344 / 9093467413) * np.cos(
                    q2 - 3357166075 / 8743247601) * np.cos(q3 + 1421104251 / 3634967671)) * np.cos(q4) + a4 * (
                -np.sin(q2 - 3357166075 / 8743247601) * np.cos(q1 + 3745578344 / 9093467413) * np.cos(
                    q3 + 1421104251 / 3634967671) - np.sin(q3 + 1421104251 / 3634967671) * np.cos(
                    q1 + 3745578344 / 9093467413) * np.cos(q2 - 3357166075 / 8743247601)) * np.sin(q4) - d2 * np.sin(
            q1 + 3745578344 / 9093467413) - d3 * np.sin(q1 + 3745578344 / 9093467413)

        y = a1 * np.sin(q1 + 3745578344 / 9093467413) + a2 * np.sin(q1 + 3745578344 / 9093467413) * np.cos(
            q2 - 3357166075 / 8743247601) - a3 * np.sin(q1 + 3745578344 / 9093467413) * np.sin(
            q2 - 3357166075 / 8743247601) * np.sin(q3 + 1421104251 / 3634967671) + a3 * np.sin(
            q1 + 3745578344 / 9093467413) * np.cos(q2 - 3357166075 / 8743247601) * np.cos(
            q3 + 1421104251 / 3634967671) + a4 * (
                -np.sin(q1 + 3745578344 / 9093467413) * np.sin(q2 - 3357166075 / 8743247601) * np.sin(
                    q3 + 1421104251 / 3634967671) + np.sin(q1 + 3745578344 / 9093467413) * np.cos(
                    q2 - 3357166075 / 8743247601) * np.cos(q3 + 1421104251 / 3634967671)) * np.cos(q4) + a4 * (
                -np.sin(q1 + 3745578344 / 9093467413) * np.sin(q2 - 3357166075 / 8743247601) * np.cos(
                    q3 + 1421104251 / 3634967671) - np.sin(q1 + 3745578344 / 9093467413) * np.sin(
                    q3 + 1421104251 / 3634967671) * np.cos(q2 - 3357166075 / 8743247601)) * np.sin(q4) + d2 * np.cos(
            q1 + 3745578344 / 9093467413) + d3 * np.cos(q1 + 3745578344 / 9093467413)

        z = -a2 * np.sin(q2 - 3357166075 / 8743247601) - a3 * np.sin(q2 - 3357166075 / 8743247601) * np.cos(
            q3 + 1421104251 / 3634967671) - a3 * np.sin(q3 + 1421104251 / 3634967671) * np.cos(
            q2 - 3357166075 / 8743247601) + a4 * (
                np.sin(q2 - 3357166075 / 8743247601) * np.sin(q3 + 1421104251 / 3634967671) - np.cos(
                    q2 - 3357166075 / 8743247601) * np.cos(q3 + 1421104251 / 3634967671)) * np.sin(q4) + a4 * (
                -np.sin(q2 - 3357166075 / 8743247601) * np.cos(q3 + 1421104251 / 3634967671) - np.sin(
                    q3 + 1421104251 / 3634967671) * np.cos(q2 - 3357166075 / 8743247601)) * np.cos(q4) + d1

        return x, y, z
