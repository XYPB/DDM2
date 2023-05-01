import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
from dipy.reconst.dti import TensorModel
from dipy.segment.mask import segment_from_cfa
from dipy.segment.mask import bounding_box
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation


def norm_data(data):
    min_data = np.min(data, axis=(0,1,2), keepdims=True)
    max_data = np.max(data, axis=(0,1,2), keepdims=True)
    return (data.astype(np.float32) - min_data) / (max_data - min_data)


class SNRMetric(object):

    def __init__(self, bvals, bvecs) -> None:
        self.gtab = gradient_table(bvals, bvecs)
        self.tenmodel = TensorModel(self.gtab)

    def eval(self, data):
        data = norm_data(data)

        _, self.mask = median_otsu(data, vol_idx=[0])
        tensorfit = self.tenmodel.fit(data)

        threshold = (0.6, 1, 0, 0.1, 0, 0.1)
        CC_box = np.zeros_like(data[..., 0])

        mins, maxs = bounding_box(self.mask)
        mins = np.array(mins)
        maxs = np.array(maxs)
        diff = (maxs - mins) // 4
        bounds_min = mins + diff
        bounds_max = maxs - diff

        CC_box[bounds_min[0]:bounds_max[0],
               bounds_min[1]:bounds_max[1],
               bounds_min[2]:bounds_max[2]] = 1

        mask_cc_part, cfa = segment_from_cfa(tensorfit, CC_box, threshold,
                                     return_cfa=True)
        mask_cc_part = mask_cc_part

        mean_signal = np.mean(data[mask_cc_part], axis=0)

        mask_noise = binary_dilation(self.mask, iterations=10)
        mask_noise[..., :mask_noise.shape[-1]//2] = 1
        mask_noise = ~mask_noise

        noise_std = np.std(data[mask_noise, :])
        return mean_signal, noise_std