from curses import raw
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import random
import os
import numpy as np
import torch
from dipy.io.image import save_nifti, load_nifti
from matplotlib import pyplot as plt
from torchvision import transforms, utils
import torchvision.transforms.functional as F

class ToTensorSeq(object):
    def __init__(self) -> None:
        pass

    def __call__(self, xs):
        assert(isinstance(xs, list))
        return [F.to_tensor(img) for img in xs]

class RandomVerticalFlipSeq(object):
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, xs):
        assert(isinstance(xs, list))
        if torch.rand(1) < self.p:
            ret = []
            for img in xs:
                ret.append(F.vflip(img))
                print(img.shape)
            return ret
        return xs


class RandomHorizontalFlipSeq(object):
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, xs):
        assert(isinstance(xs, list))
        if torch.rand(1) < self.p:
            return [F.hflip(img) for img in xs]
        return xs



class RandMRIDataset(Dataset):
    def __init__(self, dataroot, valid_mask, phase='train', image_size=128, in_channel=1, 
                 val_volume_idx=50, val_slice_idx=40, rand_sample_size=2,
                 padding=1, lr_flip=0.5, stage2_file=None, canny_path=None):
        self.padding = padding // 2
        self.lr_flip = lr_flip
        self.phase = phase
        self.in_channel = in_channel
        self.rand_sample_size = rand_sample_size

        # read data
        raw_data, _ = load_nifti(dataroot) # width, height, slices, gradients
        print('Loaded data of size:', raw_data.shape)
        # normalize data
        raw_data = raw_data.astype(np.float32) / np.max(raw_data, axis=(0,1,2), keepdims=True)

        # parse mask
        assert type(valid_mask) is (list or tuple) and len(valid_mask) == 2

        # mask data
        raw_data = raw_data[:,:,:,valid_mask[0]:valid_mask[1]] 
        self.data_size_before_padding = raw_data.shape

        self.raw_data = np.pad(raw_data, ((0,0), (0,0), (in_channel//2, in_channel//2), (self.padding, self.padding)), mode='wrap')

        if canny_path is not None:
            raw_canny, _ = load_nifti(canny_path)
            raw_canny = raw_canny.astype(np.float32) / np.max(raw_canny, axis=(0,1,2), keepdims=True)

            # mask data
            raw_canny = raw_canny[:,:,:,valid_mask[0]:valid_mask[1]] 

            self.raw_canny = np.pad(raw_canny, ((0,0), (0,0), (in_channel//2, in_channel//2), (self.padding, self.padding)), mode='wrap')
            assert(self.raw_canny.shape == self.raw_data.shape)
        else:
            self.raw_canny = None

        # running for Stage3?
        if stage2_file is not None:
            print('Parsing Stage2 matched states from the stage2 file...')
            self.matched_state = self.parse_stage2_file(stage2_file)
        else:
            self.matched_state = None

        
        # transform
        if phase == 'train':
            self.rand_transforms = transforms.Compose([
                ToTensorSeq(),
                RandomVerticalFlipSeq(lr_flip),
                RandomVerticalFlipSeq(lr_flip),
            ])
            self.transforms = transforms.Compose([
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])
        else:
            self.rand_transforms = transforms.Compose([
                ToTensorSeq(),
            ])
            self.transforms = transforms.Compose([
                transforms.Lambda(lambda t: (t * 2) - 1)
            ])

        # prepare validation data
        if val_volume_idx == 'all':
            self.val_volume_idx = range(raw_data.shape[-1])
        elif type(val_volume_idx) is int:
            self.val_volume_idx = [val_volume_idx]
        elif type(val_volume_idx) is list:
            self.val_volume_idx = val_volume_idx
        else:
            self.val_volume_idx = [int(val_volume_idx)]

        if val_slice_idx == 'all':
            self.val_slice_idx = range(0, raw_data.shape[-2])
        elif type(val_slice_idx) is int:
            self.val_slice_idx = [val_slice_idx]
        elif type(val_slice_idx) is list:
            self.val_slice_idx = val_slice_idx
        else:
            self.val_slice_idx = [int(val_slice_idx)]

    def parse_stage2_file(self, file_path):
        results = dict()
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                info = line.strip().split('_')
                volume_idx, slice_idx, t = int(info[0]), int(info[1]), int(info[2])
                if volume_idx not in results:
                    results[volume_idx] = {}
                results[volume_idx][slice_idx] = t
        return results


    def __len__(self):
        if self.phase == 'train' or self.phase == 'test':
            return self.data_size_before_padding[-2] * self.data_size_before_padding[-1] # num of volumes
        elif self.phase == 'val':
            return len(self.val_volume_idx) * len(self.val_slice_idx)

    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'test':
            # decode index to get slice idx and volume idx
            volume_idx = index // self.data_size_before_padding[-2]
            slice_idx = index % self.data_size_before_padding[-2]
        elif self.phase == 'val':
            s_index = index % len(self.val_slice_idx)
            index = index // len(self.val_slice_idx)
            slice_idx = self.val_slice_idx[s_index]
            volume_idx = self.val_volume_idx[index]

        raw_input = self.raw_data
        if self.padding > 0:
            index_in_pad = list(range(volume_idx, volume_idx+self.padding)) + list(range(volume_idx+self.padding+1, volume_idx+2*self.padding+1))
            indexes = random.sample(index_in_pad, k=self.rand_sample_size)
            sampled = [raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[idx]] for idx in indexes]
            sampled.append(raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]])
            raw_input = np.concatenate(sampled, axis=-1)
        elif self.padding == 0:
            raw_input = np.concatenate((
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding-1]],
                                    raw_input[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]), axis=-1)

        # w, h, c, d = raw_input.shape
        # raw_input = np.reshape(raw_input, (w, h, -1))
        if len(raw_input.shape) == 4:
            raw_input = raw_input[:,:,0]
        if self.raw_canny is not None:
            raw_canny_input = self.raw_canny[:,:,slice_idx:slice_idx+2*(self.in_channel//2)+1,[volume_idx+self.padding]]
            if len(raw_canny_input.shape) == 4:
                raw_canny_input = raw_canny_input[:, :, 0]
            raw_input, raw_canny_input = self.rand_transforms([raw_input, raw_canny_input])
            raw_canny_input = self.transforms(raw_canny_input)
        else:
            raw_input = self.rand_transforms([raw_input])[0]
        raw_input = self.transforms(raw_input) # only support the first channel for now
        # raw_input = raw_input.view(c, d, w, h)

        ret = dict(X=raw_input[[-1], :, :], condition=raw_input[:-1, :, :])

        if self.raw_canny is not None:
            ret['canny'] = raw_canny_input.type(torch.FloatTensor)

        if self.matched_state is not None:
            ret['matched_state'] = torch.zeros(1,) + self.matched_state[volume_idx][slice_idx]

        return ret


if __name__ == "__main__":

    # s3sh
    # valid_mask = np.zeros(193,)
    # valid_mask[1:1+64] += 1
    # valid_mask = valid_mask.astype(np.bool8)
    # dataset = MRIDataset('/media/administrator/1305D8BDB8D46DEE/stanford/sr3/scripts/data/HARDI193.nii.gz', valid_mask,
    #                      phase='train', val_volume_idx=40, padding=3)#, initial_stage_file='/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_noisemodel/stages.txt')
    
    # # hardi
    # valid_mask = np.zeros(160,)
    # valid_mask[10:] += 1
    # valid_mask = valid_mask.astype(np.bool8)
    # dataset = MRIDataset('/media/administrator/1305D8BDB8D46DEE/stanford/sr3/scripts/data/HARDI150.nii.gz', valid_mask,
    #                      phase='train', val_volume_idx=40, padding=3)#, initial_stage_file='/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_noisemodel/stages.txt')

    # gslider
    # valid_mask = np.zeros(60,)
    # valid_mask[10:] += 1
    # valid_mask = valid_mask.astype(np.bool8)
    # dataset = MRIDataset('/media/administrator/1305D8BDB8D46DEE/stanford/data/gSlider_first.nii', valid_mask,
    #                      phase='train', val_volume_idx=40, padding=3)#, initial_stage_file='/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_noisemodel/stages.txt')

    # qiyuan's data
    # valid_mask = np.zeros(108,)
    # valid_mask[18:] += 1
    # valid_mask = valid_mask.astype(np.bool8)
    valid_mask = [10,160]
    dataset = RandMRIDataset('dataset/stanford_hardi/HARDI150.nii.gz', valid_mask,
                         phase='train', val_volume_idx=40, padding=3)#, initial_stage_file='/media/administrator/1305D8BDB8D46DEE/stanford/MRI/experiments/v25_noisemodel/stages.txt')


    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for i, data in enumerate(trainloader):
        if i % 10 != 0:
            continue
        if i > 108:
            break
        img = data['X']
        condition = data['condition']
        img = img.numpy()
        condition = condition.numpy()

        vis = np.hstack((img[0].transpose(1,2,0), condition[0,[0]].transpose(1,2,0), condition[0,[1]].transpose(1,2,0)))
        # plt.imshow(img[0].transpose(1,2,0), cmap='gray')
        # plt.show()
        # plt.imshow(condition[0,[0]].transpose(1,2,0), cmap='gray')
        # plt.show()
        # plt.imshow(condition[0,[1]].transpose(1,2,0), cmap='gray')
        # plt.show()

        plt.imshow(vis, cmap='gray')
        plt.title(f'idx: {i}')
        plt.show()
        #break
