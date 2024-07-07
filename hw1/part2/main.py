import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    with open(args.setting_path,'r') as file:
        lines = file.readlines()
    
    rgbs = []
    for line in lines[1:6]:
        rgbs.append([float(v) for v in line.strip().split(',')])

    sigma_s = int(lines[6].strip().split(',')[1])
    sigma_r = float(lines[6].strip().split(',')[3])

    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)

    costs=[]
    costs.append(np.sum(abs(bf_out.astype(np.int32) - jbf_out.astype(np.int32))))
    for i, ratio in enumerate(rgbs):
        img_gray = np.dot(img_rgb[..., :3], ratio)
        cv2.imwrite(f'./save/1_gray_{ratio}.png', img_gray)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
        cv2.imwrite(f'./save/1_rgb_{ratio}.png', cv2.cvtColor(jbf_out, cv2.COLOR_BGR2RGB))
        costs.append(np.sum(abs(bf_out.astype(np.int32) - jbf_out.astype(np.int32))))

    print(costs)

if __name__ == '__main__':
    main()