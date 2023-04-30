import cv2
import os
import numpy as np
from dipy.io.image import save_nifti, load_nifti
import matplotlib.pyplot as plt
from tqdm import tqdm

def canny_detector(img, eps=1.4, weak_th=0.1, strong_th=0.4):
    img = img.squeeze()

    img = cv2.GaussianBlur(img, (5, 5,), eps)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    mag_max = np.max(mag)
    weak_th = mag_max * weak_th
    strong_th = mag_max * strong_th

    height, width = img.shape

    for i_x in range(width):
        for i_y in range(height):

            grad_ang = ang[i_y, i_x]
            grad_ang = abs(grad_ang-180) if abs(grad_ang)>180 else abs(grad_ang)

            # selecting the neighbours of the target pixel
            # according to the gradient direction
            # In the x axis direction
            if grad_ang<= 22.5:
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # top right (diagonal-1) direction
            elif grad_ang>22.5 and grad_ang<=(22.5 + 45):
                neighb_1_x, neighb_1_y = i_x-1, i_y-1
                neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

            # In y-axis direction
            elif grad_ang>(22.5 + 45) and grad_ang<=(22.5 + 90):
                neighb_1_x, neighb_1_y = i_x, i_y-1
                neighb_2_x, neighb_2_y = i_x, i_y + 1

            # top left (diagonal-2) direction
            elif grad_ang>(22.5 + 90) and grad_ang<=(22.5 + 135):
                neighb_1_x, neighb_1_y = i_x-1, i_y + 1
                neighb_2_x, neighb_2_y = i_x + 1, i_y-1

            # Now it restarts the cycle
            elif grad_ang>(22.5 + 135) and grad_ang<=(22.5 + 180):
                neighb_1_x, neighb_1_y = i_x-1, i_y
                neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
            if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
                if mag[i_y, i_x]<mag[neighb_1_y, neighb_1_x]:
                    mag[i_y, i_x]= 0
                    continue

            if width>neighb_2_x>= 0 and height>neighb_2_y>= 0:
                if mag[i_y, i_x]<mag[neighb_2_y, neighb_2_x]:
                    mag[i_y, i_x]= 0

    weak_ids = np.zeros_like(img)
    strong_ids = np.zeros_like(img)              
    ids = np.zeros_like(img)

    for i_x in range(width):
        for i_y in range(height):
            grad_mag = mag[i_y, i_x]
            if grad_mag<weak_th:
                mag[i_y, i_x]= 0
            elif strong_th>grad_mag>= weak_th:
                ids[i_y, i_x]= 1
            else:
                ids[i_y, i_x]= 2

    # finally returning the magnitude of
    # gradients of edges
    return mag

if __name__ == '__main__':
    raw_data, _ = load_nifti('dataset/sherbrooke_3shell/HARDI193.nii.gz')
    dest = './tmp/s3sh_canny'
    eps = 2.5
    # raw_data, _ = load_nifti('dataset/stanford_hardi/HARDI150.nii.gz')
    # dest = './tmp/hardi_canny'
    # eps = 2
    os.makedirs(dest, exist_ok=True)
    _, _, slices, volumes = raw_data.shape
    print(raw_data.shape)
    weak_th_side = np.linspace(0.2, 0.45, volumes)
    weak_th_mid = np.linspace(0.2, 0.4, volumes)

    canny_imgs = []
    for i in tqdm(range(slices)):
        slice_imgs = []
        for j in range(volumes):
            if i <= slices * 0.15 or i >= slices * 0.85:
                weak_th = weak_th_side[j]
            else:
                weak_th = weak_th_mid[j]
            img = raw_data[:, :, i, j]
            canny = canny_detector(img, eps=2, weak_th=weak_th)
            plt.imsave(os.path.join(dest, f'canny_{i}_{j}_orig.png'), img, cmap='gray')
            plt.imsave(os.path.join(dest, f'canny_{i}_{j}.png'), canny, cmap='gray')
            slice_imgs.append(canny)
        slice_imgs = np.stack(slice_imgs, axis=-1)
        canny_imgs.append(slice_imgs)
    canny_imgs = np.stack(canny_imgs, axis=-2)
    print(canny_imgs.shape)

    save_nifti('dataset/sherbrooke_3shell/HARDI193_canny.nii.gz', canny_imgs, affine=np.eye(4))
    # save_nifti('dataset/stanford_hardi/HARDI150_canny.nii.gz', canny_imgs, affine=np.eye(4))
