import dipy.reconst.dti as dti
import dipy.reconst.csdeconv as csd
import numpy as np
from dipy.segment.mask import median_otsu
import dipy.reconst.cross_validation as xval
import copy
import scipy.stats as stats
import os
from joblib import Parallel, delayed # this is for parallelization
from matplotlib import pyplot as plt
from dipy.io.image import save_nifti, load_nifti
import dipy.data as dpd
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
import pandas as pd


class MRIMetrics():
    def __init__(self, gtab):
        self.gtab = gtab
        self.dti_model = dti.TensorModel(gtab)

    def fit_model(self, data):
        #response, ratio = csd.auto_response_ssst(self.gtab, data, roi_radius=10, fa_thr=0.7)
        response, ratio = csd.auto_response_ssst(self.gtab, data, roi_radii=10, fa_thr=0.7)
        csd_model = csd.ConstrainedSphericalDeconvModel(self.gtab, response)
        return csd_model, response

    def pearsonr(self, data, dti):
        return stats.pearsonr(data, dti)[0] ** 2

    def eval(self, data_slice, csd_model, response):

        dti_slice = xval.kfold_xval(self.dti_model, data_slice, 2)
        csd_slice = xval.kfold_xval(csd_model, data_slice, 2, response)
        print(data_slice.shape, dti_slice.shape)

        r2s_dti = []
        for i in range(0, dti_slice.shape[0]):
            for j in range(0, dti_slice.shape[1]):
                for k in range(0, dti_slice.shape[2]):
                    dti_r2 = stats.pearsonr(data_slice[i, j, k], dti_slice[i, j, k])[0] ** 2
                    r2s_dti.append(dti_r2)

        # r2s_dti = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], dti_slice[i, j, k]) for i in range(0, dti_slice.shape[0]) for j in range(0, dti_slice.shape[1]) for k in range(0, dti_slice.shape[2]))


        r2s_dti = np.array(r2s_dti)
        r2s_dti = r2s_dti[~np.isnan(r2s_dti)]

        r2s_csd = []
        for i in range(0, csd_slice.shape[0]):
            for j in range(0, csd_slice.shape[1]):
                for k in range(0, csd_slice.shape[2]):
                    csd_r2_mp = stats.pearsonr(data_slice[i, j, k], csd_slice[i, j, k])[0] ** 2

        # r2s_csd = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], csd_slice[i, j, k]) for i in range(0, csd_slice.shape[0]) for j in range(0, csd_slice.shape[1]) for k in range(0, csd_slice.shape[2]))

        r2s_csd = np.array(r2s_csd)
        r2s_csd = r2s_csd[~np.isnan(r2s_csd)]

        return dti_slice, csd_slice

    def calc(self, data, slice=38):
        csd_model, response = self.fit_model(data)
        
        # mask with otsu
        _, mask = median_otsu(data, vol_idx=[0, 1])
        data_masked = copy.deepcopy(data)
        
        if slice is not None:
            data_masked = data_masked[..., slice:slice+1, :]
            data_masked[mask[..., slice:slice+1] == 0] = 0

        dti, csd = self.eval(data_masked, csd_model, response)

        return dti, csd


class DTIMetrics():
    def __init__(self, gtab):
        self.gtab = gtab
        self.dti_model = dti.TensorModel(gtab)

    def eval(self, data_slice):

        dti_slice = xval.kfold_xval(self.dti_model, data_slice, 2)

        r2s_dti = []
        for i in range(0, dti_slice.shape[0]):
            for j in range(0, dti_slice.shape[1]):
                for k in range(0, dti_slice.shape[2]):
                    dti_r2 = stats.pearsonr(data_slice[i, j, k], dti_slice[i, j, k])[0] ** 2
                    r2s_dti.append(dti_r2)

        # r2s_dti = Parallel(n_jobs=8)(delayed(self.pearsonr)(data_slice[i, j, k], dti_slice[i, j, k]) for i in range(0, dti_slice.shape[0]) for j in range(0, dti_slice.shape[1]) for k in range(0, dti_slice.shape[2]))

        r2s_dti = np.array(r2s_dti)
        r2s_dti = r2s_dti[~np.isnan(r2s_dti)]

        return dti_slice

    def calc(self, data, slice=38):
        # mask with otsu
        # _, mask = median_otsu(data, vol_idx=[0, 1])
        # data_masked = copy.deepcopy(data)
        return self.eval(data[..., slice:slice+1, :])
        


def plot(data_dti, ddm_dti, p2s_dti, our_dti):
    import seaborn as sns
    from statannot import add_stat_annotation

    df_diff = pd.DataFrame({'(DDM) DTI':ddm_dti - data_dti,
                        '(P2S) DTI':p2s_dti - data_dti,
                        '(Our) DTI':our_dti - data_dti})
    
    
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="variable", y="value", data=pd.melt(df_diff), fliersize=0, sym='', palette="Set2")

    
    add_stat_annotation(ax, data=pd.melt(df_diff), x="variable", y="value",
                        box_pairs=[('(DDM - Noisy) DTI', '(P2S - Noisy) DTI', '(Our - Noisy) DTI')],
                                test='t-test_ind', text_format='star', loc='outside', verbose=2)

if __name__ == '__main__':
    # loading gtab
    # data_root = 'dataset/stanford_hardi/HARDI150.nii.gz'
    # _, gtab = dpd.read_stanford_hardi()
    # mask = [10, 160]
    data_root = 'dataset/sherbrooke_3shell/HARDI193.nii.gz'
    _, gtab = dpd.read_sherbrooke_3shell()
    mask = [65,129]

    bvals = gtab.bvals
    bvecs = gtab.bvecs

    # bvals = bvals[mask[0]:mask[1]]
    # bvecs = bvecs[mask[0]:mask[1]]

    sel_b = np.logical_or(bvals == 0, bvals == 2000)

    gtab = gradient_table(bvals[sel_b], bvecs[sel_b])

    # loading original datla
    data, _ = load_nifti(data_root)
    # data = data[..., mask[0]:mask[1]]

    #data = data.astype(np.float32) / max_data
    data = data[..., sel_b]
    max_data = np.max(data, axis=(0,1,2), keepdims=True)

    # loading our data
    stage1, _ = load_nifti('experiments/s3sh_denoise_230430_045617_ddim/results/s3sh_denoised.nii.gz')
    print(data.shape, stage1.shape)
    data_ours = np.concatenate((data[:,:,:,[0]], stage1), axis=-1)
    data_ours[:,:,:,1:] = data_ours[:,:,:,1:] * max_data[:,:,:,1:]

    # loading p2s
    data_p2s, _ = load_nifti('experiments/s3sh_p2s/denoised_sherbrooke_3shell_p2s_mlp.nii.gz')
    # data_p2s = data_p2s[..., mask[0]:mask[1]]
    #data_p2s = data_p2s.astype(np.float32) / max_data
    data_p2s[:,:,:,0] = data[:,:,:,0]
    data_p2s = data_p2s[..., sel_b]

    # loading ddm
    data_ddm, _ = load_nifti('experiments/s3sh_denoise_230426_233927_baseline/results/s3sh_denoised.nii.gz')
    #data_ddm = data_ddm.astype(np.float32) / max_data
    data_ddm = np.concatenate((data[:,:,:,[0]], data_ddm), axis=-1)
    data_ddm[:,:,:,1:] = data_ddm[:,:,:,1:][..., sel_b[mask[0]:mask[1]]]


    # DTI calculation
    M = DTIMetrics(gtab)

    dti_raw = M.calc(data, slice=38)

    print(data_ddm.shape, data.shape)
    dti_ddm = M.calc(data_ddm, slice=38)

    print('ddm:', np.mean(dti_ddm - dti_raw))

    dti_p2s = M.calc(data_p2s, slice=38)

    print('P2S:', np.mean(dti_p2s - dti_raw))

    dti_ours = M.calc(data_ours, slice=38)

    print('Ours:', np.mean(dti_ours - dti_raw))

    # plot
    plot(dti_raw, dti_ddm, dti_p2s, dti_ours)


