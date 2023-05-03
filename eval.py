import numpy as np
import copy
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.segment.mask import median_otsu
import dipy.reconst.cross_validation as xval
import dipy.reconst.dti as DTI
import dipy.reconst.csdeconv as CSD
import scipy.stats as stats
from multiprocessing import Pool


def norm_data(data):
    min_data = np.min(data, axis=(0,1,2), keepdims=True)
    max_data = np.max(data, axis=(0,1,2), keepdims=True)
    return (data.astype(np.float32) - min_data) / (max_data - min_data)


class MRIMetrics():
    def __init__(self, bvals, bvecs, raw_data):
        self.gtab = self.gtab = gradient_table(bvals, bvecs)
        self.dti_model = DTI.TensorModel(self.gtab)
        self.raw_data = raw_data
        self.raw_data_slice = None
        _, self.mask = median_otsu(raw_data, vol_idx=[0, 1])

    def fit_model(self, data):
        response, ratio = CSD.auto_response_ssst(self.gtab, data, roi_radii=10, fa_thr=0.7)
        csd_model = CSD.ConstrainedSphericalDeconvModel(self.gtab, response)
        return csd_model, response

    def pearsonr(self, data, dti):
        return stats.pearsonr(data, dti)[0] ** 2

    def eval(self, data_slice, csd_model, response):

        dti_slice = xval.kfold_xval(self.dti_model, data_slice, 2)
        csd_slice = xval.kfold_xval(csd_model, data_slice, 2, response)
        print(self.raw_data_slice.shape, dti_slice.shape)

        r2s_dti = []
        for i in range(0, dti_slice.shape[0]):
            for j in range(0, dti_slice.shape[1]):
                for k in range(0, dti_slice.shape[2]):
                    dti_r2 = stats.pearsonr(data_slice[i, j, k], dti_slice[i, j, k])[0] ** 2
                    r2s_dti.append(dti_r2)
        r2s_dti = np.array(r2s_dti)
        r2s_dti = r2s_dti[~np.isnan(r2s_dti)]

        r2s_csd = []
        for i in range(0, csd_slice.shape[0]):
            for j in range(0, csd_slice.shape[1]):
                for k in range(0, csd_slice.shape[2]):
                    csd_r2 = stats.pearsonr(data_slice[i, j, k], csd_slice[i, j, k])[0] ** 2
                    r2s_csd.append(csd_r2)
        r2s_csd = np.array(r2s_csd)
        r2s_csd = r2s_csd[~np.isnan(r2s_csd)]

        return r2s_dti, r2s_csd

    def calc(self, data, slice=38):
        print(data.shape)
        csd_model, response = self.fit_model(data)

        data_masked = copy.deepcopy(data)

        if slice is not None:
            data_masked = data_masked[..., slice:slice+1, :]
            data_masked[self.mask[..., slice:slice+1] == 0] = 0
            self.raw_data_slice = self.raw_data[..., slice:slice+1, :]
            self.raw_data_slice[self.mask[..., slice:slice+1] == 0] = 0

        dti, csd = self.eval(data_masked, csd_model, response)

        return dti, csd
    

def get_dti_csd_score(data, bvals, bvecs, data_path, local_slice=None):
    M = MRIMetrics(bvals, bvecs, data)
    print(data_path.replace('.nii.gz', '_dti.npy'))

    our_data, _ = load_nifti(data_path)
    our_data = norm_data(our_data)
    our_data = np.concatenate([data[..., [0]], our_data], axis=-1)

    our_dti, our_csd = M.calc(our_data, slice=local_slice)

    np.save(data_path.replace('.nii.gz', '_dti.npy'), our_dti)
    np.save(data_path.replace('.nii.gz', '_csd.npy'), our_csd)


if __name__ == '__main__':
    hardi_fname = 'dataset/stanford_hardi/HARDI150.nii.gz'
    hardi_bval_fname = 'dataset/stanford_hardi/HARDI150.bval'
    hardi_bvec_fname = 'dataset/stanford_hardi/HARDI150.bvec'

    data, affine = load_nifti(hardi_fname)
    data = norm_data(data)
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    voxel_mask = [0] + list(range(10, 160))
    # voxel_mask = list(range(10, 160))
    data = data[..., voxel_mask]
    bvals = bvals[voxel_mask]
    bvecs = bvecs[voxel_mask]

    data_path = [
        'experiments/hardi150_denoise_230430_115119_control/results/hardi150_denoised.nii.gz',
        'experiments/hardi150_denoise_230426_211632_baseline/results/hardi150_denoised.nii.gz',
        'experiments/hardi150_denoise_230430_033002_ddim/results/hardi150_denoised.nii.gz',
        'experiments/hardi150_p2s/denoised_StanfordHardi_p2s_mlp.nii.gz',
    ]

    args = [(data, bvals, bvecs, d) for d in data_path]

    with Pool(4) as p:
        p.map(get_dti_csd_score, args)



