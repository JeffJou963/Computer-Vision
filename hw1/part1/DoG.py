import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_gaussian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        h,w = image.shape # (250,350)
        
        for i in range(self.num_octaves):
            gaussian_images.append(image)
            for j in range(1, self.num_gaussian_images_per_octave):
                gaussian_images.append(cv2.GaussianBlur(image, (0,0), sigmaX = self.sigma**j, sigmaY = self.sigma**j))
            image = cv2.resize(gaussian_images[-1], (w//2,h//2), interpolation = cv2.INTER_NEAREST)


        # for i in range(len(gaussian_images)):
        #     cv2.imwrite(f'./save/img{i}.png',gaussian_images[i])

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                dog_images.append(cv2.subtract(gaussian_images[5*i+j+1],gaussian_images[5*i+j]))

        # for i in range(len(dog_images)):
        #     dog_img = dog_images[i]
        #     max = np.max(dog_img)
        #     min = np.min(dog_img)
        #     norm_dog = 255 * (dog_img - min) / (max - min)
        #     cv2.imwrite(f'./save/norm_DoG{i}.png', norm_dog)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):
             for j in range(1 ,self.num_DoG_images_per_octave-1): # 1,2,5,6
                for x in range(1, (w//(i+1))-1):
                    for y in range(1, (h//(i+1))-1):
                        z = 4*i + j
                        center = dog_images[z][y,x]
                        neighbors =[]

                        for dz in [-1,0,1]:
                            for dy in [-1,0,1]:
                                for dx in [-1,0,1]:
                                    if (dz==0 and dy==0 and dx ==0):
                                        continue
                                    neighbors.append(dog_images[z+dz][(y+dy), (x+dx)])

                        if center > np.max(neighbors) or center < np.min(neighbors):
                            if abs(center) > self.threshold:
                                if i==0:
                                    keypoints.append((y,x))
                                else:
                                    keypoints.append((2*y, 2*x))
        
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
