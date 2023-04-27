import sys
sys.path.append("..")

import numpy as np
import time
import matplotlib.pyplot as plt
from model.patch2self import patch2self
from dipy.io.image import save_nifti, load_nifti
from dipy.data import read_stanford_labels

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
affine = np.eye(4)

def plot(data, denoised_sh_mlp):
    sli = 35
    vol = 127
    orig1 = data[:, :, sli, vol]

    den1 = denoised_sh_mlp[:, :, sli, vol]
    rms_diff1 = np.sqrt((orig1 - den1) ** 2)

    fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    ax[0].imshow(orig1.T, cmap='gray', origin='lower', interpolation='None')
    ax[0].set_title('Original')
    ax[0].set_axis_off()
    ax[1].imshow(den1.T, cmap='gray', origin='lower', interpolation='None')
    ax[1].set_title('Patch2Self')
    ax[1].set_axis_off()
    ax[2].imshow(rms_diff1.T, cmap='gray', origin='lower', interpolation='None')
    ax[2].set_title('Residual')
    ax[2].set_axis_off()
    plt.show()

if __name__ == '__main__':
    t1 = time.time()
    denoised_sh_mlp = patch2self(data, model='mlp')
    t2 = time.time()
    print('Time Taken: ', t2-t1)

    save_nifti('denoised_StanfordHardi_p2s_mlp.nii.gz', denoised_sh_mlp, affine)