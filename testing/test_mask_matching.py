import cv2 as cv
import numpy as np
import math
from modules.fabric_adjustor_prior import _match_masks

def create_dummy_mask(size=(128, 128)):
    mask = np.zeros(size, dtype=np.uint8)
    # Draw an L-shape (non-symmetric) to easily track rotation and translation
    cv.rectangle(mask, (40, 30), (60, 90), 255, -1)
    cv.rectangle(mask, (40, 70), (90, 90), 255, -1)
    return mask

def transform_mask(mask, tx, ty, theta_deg):
    rows, cols = mask.shape
    # Center of rotation
    center = (cols / 2, rows / 2)
    # Get rotation matrix (rotate by theta_deg)
    M_rot = cv.getRotationMatrix2D(center, theta_deg, 1.0)
    
    # Add translation to the matrix
    M_rot[0, 2] += tx
    M_rot[1, 2] += ty
    
    warped = cv.warpAffine(mask, M_rot, (cols, rows))
    return warped

def run_tests():
    prior_mask = create_dummy_mask()
    
    # Applied transform to simulate a moved fabric
    # (The fabric has moved +10 in X, -5 in Y, and rotated +15 degrees)
    applied_tx = 10.0
    applied_ty = -5.0
    applied_theta_deg = 15.0
    
    current_mask = transform_mask(prior_mask, applied_tx, applied_ty, applied_theta_deg)
    
    print(f"Applied Offset: tx={applied_tx}, ty={applied_ty}, rot={applied_theta_deg} deg")
    print("-" * 50)
    
    # Test Moments
    trans_m, rot_m = _match_masks(prior_mask, current_mask, method='moments')
    print("Method: Moments")
    print(f"Calculated Correction -> tx: {trans_m[0]:.2f}, ty: {trans_m[1]:.2f}, rot: {np.degrees(rot_m[0]):.2f} deg")
    
    # Test ECC
    trans_ecc, rot_ecc = _match_masks(prior_mask, current_mask, method='ecc')
    print("Method: ECC")
    print(f"Calculated Correction -> tx: {trans_ecc[0]:.2f}, ty: {trans_ecc[1]:.2f}, rot: {np.degrees(rot_ecc[0]):.2f} deg")

if __name__ == "__main__":
    run_tests()
