import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        key_point1, descriptor1 = orb.detectAndCompute(im1, None)
        key_point2, descriptor2 = orb.detectAndCompute(im2, None)
        # len(matches) = 161
        matches = matcher.match(descriptor1, descriptor2)
        matches = sorted(matches, key=lambda x:x.distance)

        pt1 = []
        pt2 = []
        for match in matches:
            pt1.append(key_point1[match.queryIdx].pt)
            pt2.append(key_point2[match.trainIdx].pt)

        pt1, pt2 = np.array(pt1), np.array(pt2)
        # pt1.shape = (161, 3)
        pt1, pt2 = np.hstack((pt1, np.ones((pt1.shape[0],1), dtype=int))), np.hstack((pt2, np.ones((pt2.shape[0],1), dtype=int)))

        num_matches = len(matches)
        prev_total = 0
        # num_iter = 1000
        # threshold = 6
        num_iter = 1000
        threshold = 6
        # TODO: 2. apply RANSAC to choose best H
        for iter in range(num_iter):
            rand = random.sample(range(num_matches), 4)
            p1, p2 = pt1[rand], pt2[rand]
            H_current = solve_homography(p2, p1)
            p2_H = p2 @ H_current.T
            p2_H /= p2_H[:,2:3]

            pt2_H = pt2 @ H_current.T
            x2_H, y2_H = pt2_H[:,0]/pt2_H[:,2], pt2_H[:,1]/pt2_H[:,2]
            x1, y1 = pt1[:,0], pt1[:,1]
            distance = np.sqrt((x1 - x2_H)**2 + (y1 - y2_H)**2)
            # print(distance)

            total_inliers = np.sum(distance < threshold)
            
            if (iter == 0) or (total_inliers > prev_total):
                H_best = H_current
                prev_total = total_inliers
                print(f'iter:{iter} total inliers = {prev_total}')

        # TODO: 3. chain the homographies``
        if idx == 0:
            last_best_H = H_best
        else:
            last_best_H = last_best_H @ H_best

        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
