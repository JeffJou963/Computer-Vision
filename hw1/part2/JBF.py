
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        # (300,400,3)->(312,412,3)
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        h, w = img.shape[0], img.shape[1]
        output = np.zeros_like(img)

        # Spatial kernel
        x, y = np.meshgrid(np.arange(self.wndw_size) - self.pad_w, np.arange(self.wndw_size) - self.pad_w)
        # x, y = np.meshgrid(np.arange(w) -w//2, np.arange(h) - h//2)
        kernel_s = np.exp(-0.5 * (x ** 2 + y ** 2) / (self.sigma_s ** 2))
        # Range kernel lookup table
        LUT = np.exp(-0.5 * np.arange(256) * np.arange(256) / ( (self.sigma_r * 255) ** 2))


        for y in range(self.wndw_size):
            for x in range(self.wndw_size):
                if padded_guidance.ndim == 2:
                    weight = LUT[abs(padded_guidance[y:y + h, x:x + w] - padded_guidance[y+self.pad_w, x+self.pad_w])] * kernel_s
                else:
                    weight = LUT[abs(padded_guidance[y:y + h, x:x + w, 0] - padded_guidance[y+self.pad_w, x+self.pad_w, 0])] * \
                             LUT[abs(padded_guidance[y:y + h, x:x + w, 1] - padded_guidance[y+self.pad_w, x+self.pad_w, 1])] * \
                             LUT[abs(padded_guidance[y:y + h, x:x + w, 2] - padded_guidance[y+self.pad_w, x+self.pad_w, 2])] * kernel_s
                w_accumulation = np.sum(weight)
                output[y, x, 0] = np.sum(weight * padded_img[y:y + h, x:x + w, 0]) / w_accumulation
                output[y, x, 1] = np.sum(weight * padded_img[y:y + h, x:x + w, 1]) / w_accumulation
                output[y, x, 2] = np.sum(weight * padded_img[y:y + h, x:x + w, 2]) / w_accumulation

        return np.clip(output, 0, 255).astype(np.uint8)
