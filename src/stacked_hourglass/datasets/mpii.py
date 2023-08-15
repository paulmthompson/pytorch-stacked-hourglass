import gzip
import json
import os
import random
import functools

import numpy as np
import torch
import torch.utils.data as data
from importlib_resources import open_binary
from scipy.io import loadmat
from tabulate import tabulate

import stacked_hourglass.res
from stacked_hourglass.datasets.common import DataInfo
from stacked_hourglass.utils.imutils import load_image, draw_labelmap
from stacked_hourglass.utils.misc import to_torch
from stacked_hourglass.utils.disk import getCache
from stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform

MPII_JOINT_NAMES = [
    'right_ankle', 'right_knee', 'right_hip', 'left_hip',
    'left_knee', 'left_ankle', 'pelvis', 'spine',
    'neck', 'head_top', 'right_wrist', 'right_elbow',
    'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist'
]

raw_cache = getCache('MPII')

class Mpii(data.Dataset):
    DATA_INFO = DataInfo(
        rgb_mean=[0.4404, 0.4440, 0.4327],
        rgb_stddev=[0.2458, 0.2410, 0.2468],
        joint_names=MPII_JOINT_NAMES,
        hflip_indices=[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10],
    )

    # Suggested joints to use for average PCK calculations.
    ACC_JOINTS = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]

    # The ratio between input spatial resolution vs. output heatmap spatial resolution
    INPUT_OUTPUT_RATIO = 4

    def __init__(self, image_path, is_train=True, inp_res=256, sigma=1, scale_factor=0.25,
                 rot_factor=30, label_type='Gaussian', num=0):
        self.img_folder = image_path # root image folders
        self.is_train = is_train # training set or test set
        if not isinstance(inp_res, (list, tuple)):  # Input res stored as (H, W)
            self.inp_res = [inp_res, inp_res]
        else:
            self.inp_res = inp_res
        self.out_res = [int(self.inp_res[0] / self.INPUT_OUTPUT_RATIO),
                        int(self.inp_res[1] / self.INPUT_OUTPUT_RATIO)]
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        # create train/val split

        self.anno = getAnnotations()

        self.train_list, self.valid_list = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid_list.append(idx)
            else:
                self.train_list.append(idx)
        
        if (num > 0):
            print('Using partial dataset with ', num, ' images')
            if self.is_train == False:
                self.valid_list = self.valid_list[0:num]
            else:
                self.train_list = self.train_list[0:num]
            

    def __getitem__(self, index_input):
        
        #Each of our MPII items will be centered and scaled slightly at baseline, and then may be further scaled and rotated during augmentation
        #We can cause the initial scale and augmentation.
        
        sf = self.scale_factor
        rf = self.rot_factor
        
        if self.is_train:
            index = self.train_list[index_input]
        else:
            index = self.valid_list[index_input]
        
        #img loads image, converts to float32, and converts to a tensor
        img = getImage(index,self.img_folder,self.inp_res)  # CxHxW
        
        target, target_weight, tpts = getLabelHeatmap(index,self.out_res,self.sigma,self.label_type)
        
        pts, c, s = getKeypoints(index)
        
        r = 0
        if self.is_train:
            #s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            #r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            #if random.random() <= 0.5:
                #img = fliplr(img)
                #pts = shufflelr(pts, img.size(2), self.DATA_INFO.hflip_indices)
                #c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        #inp = crop(img, c, s, self.inp_res, rot=r)
        inp = color_normalize(img, self.DATA_INFO.rgb_mean, self.DATA_INFO.rgb_stddev)

        # Meta info
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s)

        meta = {'index' : index_input, 'center' : c, 'scale' : s,
        'pts' : pts, 'tpts' : tpts, 'target_weight': target_weight}

        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)
        

@raw_cache.memoize(typed=True)
def getLabelHeatmap(index,out_res,sigma,label_type):
                                                
    pts,c,s = getKeypoints(index)                                            

    nparts = pts.size(0)
    # Generate ground truth
    tpts = pts.clone()
    target = torch.zeros(nparts, *out_res)
    target_weight = tpts[:, 2].clone().view(nparts, 1)
        
    for i in range(nparts):
        if tpts[i, 1] > 0:
            tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, out_res))
            target[i], vis = draw_labelmap(target[i], tpts[i]-1, sigma, type=label_type)
            target_weight[i, 0] *= vis
                                                
    return target, target_weight, tpts

@raw_cache.memoize(typed=True)
def getImage(index,img_folder,inp_res):
        
    anno = getAnnotations()
    
    a = anno[index]
    
    img_path = os.path.join(img_folder, a['img_paths'])
    
    (pts,c,s) = getKeypoints(index)
    
    image = load_image(img_path)
    inp = crop(image, c, s, inp_res)
    return inp

def getKeypoints(index):
    
    anno = getAnnotations()
    
    a = anno[index]
    c = torch.Tensor(a['objpos'])
    s = a['scale_provided']
    
    pts = torch.Tensor(a['joint_self'])
    
    # Adjust center/scale slightly to avoid cropping limbs
    if c[0] != -1:
        c[1] = c[1] + 15 * s
        s = s * 1.25
    
    return pts,c,s

@functools.lru_cache()
def getAnnotations():
    with gzip.open(open_binary(stacked_hourglass.res, 'mpii_annotations.json.gz')) as f:
        anno = json.load(f)
    return anno

def evaluate_mpii_validation_accuracy(preds):
    threshold = 0.5
    SC_BIAS = 0.6

    dict = loadmat(open_binary(stacked_hourglass.res, 'detections_our_format.mat'))
    jnt_missing = dict['jnt_missing']
    pos_gt_src = dict['pos_gt_src']
    headboxes_src = dict['headboxes_src']

    preds = np.array(preds)
    assert preds.shape == (pos_gt_src.shape[2], pos_gt_src.shape[0], pos_gt_src.shape[1])
    pos_pred_src = np.transpose(preds, [1, 2, 0])

    jnt_visible = 1 - jnt_missing
    uv_error = pos_pred_src - pos_gt_src
    uv_err = np.linalg.norm(uv_error, axis=1)
    headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    headsizes = np.linalg.norm(headsizes, axis=0)
    headsizes *= SC_BIAS
    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)
    less_than_threshold = np.multiply((scaled_uv_err < threshold), jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)

    PCKh = np.ma.array(PCKh, mask=False)
    PCKh.mask[6:8] = True

    return PCKh


def print_mpii_validation_accuracy(preds):
    PCKh = evaluate_mpii_validation_accuracy(preds)

    joint_names = Mpii.DATA_INFO.joint_names

    head = joint_names.index('head_top')
    lsho = joint_names.index('left_shoulder')
    lelb = joint_names.index('left_elbow')
    lwri = joint_names.index('left_wrist')
    lhip = joint_names.index('left_hip')
    lkne = joint_names.index('left_knee')
    lank = joint_names.index('left_ankle')
    rsho = joint_names.index('right_shoulder')
    relb = joint_names.index('right_elbow')
    rwri = joint_names.index('right_wrist')
    rkne = joint_names.index('right_knee')
    rank = joint_names.index('right_ankle')
    rhip = joint_names.index('right_hip')

    print(tabulate([
        ['Head', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean'],
        [PCKh[head], 0.5 * (PCKh[lsho] + PCKh[rsho]), 0.5 * (PCKh[lelb] + PCKh[relb]),
        0.5 * (PCKh[lwri] + PCKh[rwri]), 0.5 * (PCKh[lhip] + PCKh[rhip]),
        0.5 * (PCKh[lkne] + PCKh[rkne]), 0.5 * (PCKh[lank] + PCKh[rank]), np.mean(PCKh)]
    ], headers='firstrow', floatfmt='0.2f'))
