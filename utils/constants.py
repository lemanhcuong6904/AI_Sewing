import numpy as np

# ---------------------------------------------------------------------------
# Global physical coordinates and vectors
# ---------------------------------------------------------------------------

# The default origin (corner) of the fabric in the camera frame (pixels)
DEFAULT_X_BASIS = 251 # 446
DEFAULT_Y_BASIS = 200

U1_VECTOR = np.array([1, 0], dtype=np.float64)    # Horizontal unit vector
U2_VECTOR = np.array([0, -1], dtype=np.float64)   # Vertical unit vector

# Scale conversions
PX_PER_CM_X = 39.0                   # 1 cm = 39 pixels horizontally
PX_PER_CM_Y = 35.0                   # 1 cm = 32 pixels vertically

# Use X scale for generic baseline definitions
AXIS_LENGTH_PX = PX_PER_CM_X * 12    # default axis length: 12 cm = 468 pixels

# Deadband thresholds (cm/deg) - Error below this is ignored (fabric is stable)
TRANSLATION_DEADBAND_CM = 0.03
ROTATION_DEADBAND_DEG = 0.1

# Default robot hand center of rotation (pivot point) in camera pixels
DEFAULT_PIVOT_X = 545.0
DEFAULT_PIVOT_Y = -370.0
DEFAULT_PIVOT = (DEFAULT_PIVOT_X, DEFAULT_PIVOT_Y)
