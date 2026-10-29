import numpy as np
import math

def signed_angle_between_deg(v, u):
    theta_v = np.arctan2(v[1], v[0])
    theta_u = np.arctan2(u[1], u[0])
    angle = theta_v - theta_u
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if angle > np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi
    return math.degrees(angle)

u1 = np.array([-1, 0])
# Clockwise rotation by 18 degrees (Right to Down)
# So vector pointing Left [-1, 0] rotates to point Left-Up? No, Left-Up is [-cos, -sin]
# Wait, in image coords, Clockwise rotates X->Y (Right->Down).
# So [-1, 0] (Left) rotates to [ -cos(18), -sin(18) ]
v_cw = np.array([-math.cos(math.radians(18)), -math.sin(math.radians(18))])
print("CW rotation angle:", signed_angle_between_deg(v_cw, u1))

v_ccw = np.array([-math.cos(math.radians(18)), math.sin(math.radians(18))])
print("CCW rotation angle:", signed_angle_between_deg(v_ccw, u1))

# What if line points Right?
v_cw_right = np.array([math.cos(math.radians(18)), math.sin(math.radians(18))])
print("CW rotation angle (pointing right):", signed_angle_between_deg(v_cw_right, u1))
