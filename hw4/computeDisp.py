import numpy as np
import cv2.ximgproc as xip
import cv2

def census_loss(h, w, img):
    census = np.zeros((h, w), dtype=np.uint8)
    center = img[1:h+1, 1:w+1]
    offsets = [(u,v) for v in range(3) for u in range(3) if not u == 1 == v]

    for u,v in offsets:
        census = (census << 1) | (img[v:v+h, u:u+w] >= center)
    return census

def computeDisp(Il, Ir, max_disp):

    # (375, 450)
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    Il_g = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_g = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    Il_p = cv2.copyMakeBorder(Il_g, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    Ir_p = cv2.copyMakeBorder(Ir_g, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    census_l = census_loss(h, w, Il_p)
    census_r = census_loss(h, w, Ir_p)

    cost_l = np.full((h, w, max_disp), 24, dtype=np.float32)
    cost_r = np.full((h, w, max_disp), 24, dtype=np.float32)

    for d in range(max_disp):
        if d == 0:
            shift_r = census_r
            shift_l = census_l
        else:
            shift_r = np.zeros_like(census_r)
            shift_r[:, d:] = census_r[:, :-d]
            shift_l = np.zeros_like(census_l)
            shift_l[:, :-d] = census_l[:, d:]

        xor_l = census_l ^ shift_r
        xor_r = census_r ^ shift_l

        cost_l[:,:,d] = np.vectorize(lambda x: bin(x).count('1'))(xor_l)
        cost_r[:,:,d] = np.vectorize(lambda x: bin(x).count('1'))(xor_r)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    wdnw_size = 5
    sigma_r = 7
    sigma_s = 13
    for d in range(max_disp):
        cost_l[:,:,d] = xip.jointBilateralFilter(Il_g, cost_l[:,:,d], wdnw_size, sigma_r, sigma_s)
        cost_r[:,:,d] = xip.jointBilateralFilter(Ir_g, cost_r[:,:,d], wdnw_size, sigma_r, sigma_s)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    disparity_l = np.argmin(cost_l, axis=2)
    disparity_r = np.argmin(cost_r, axis=2)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    x, y = np.meshgrid(range(w), range(h))
    valid = (x - disparity_l >= 0) & (disparity_r[y, x - disparity_l] == disparity_l)
    consistency = np.where(valid, disparity_l, -1)

    # Hole filling
    consistency = cv2.copyMakeBorder(consistency, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=max_disp)
    FL = np.zeros((h, w), dtype=np.float32)
    FR = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            idx_L, idx_R = 0, 0
            while consistency[y, x+1-idx_L] == -1:
                idx_L += 1
            FL[y, x] = consistency[y, x+1-idx_L]
            while consistency[y, x+1+idx_R] == -1:
                idx_R += 1
            FR[y, x] = consistency[y, x+1+idx_R]
    labels = np.min((FL, FR), axis=0)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, r=11)

    return labels.astype(np.uint8)