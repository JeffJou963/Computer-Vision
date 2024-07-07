import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=1.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float64)

    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    keypoints = DoG.get_keypoints(img)
    save_path = f'./save/result2_thres{args.threshold}.png'
    plot_keypoints(img, keypoints,save_path)

    # keypoints_gt = np.load('./testdata/1_gt.npy')
    # plot_keypoints(img, keypoints_gt,'./save/gt.png')



if __name__ == '__main__':
    main()