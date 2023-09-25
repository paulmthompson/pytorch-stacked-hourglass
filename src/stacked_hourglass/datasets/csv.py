import csv
import os
import random
import functools

import numpy as np
import torch
import torch.utils.data as data
from importlib_resources import open_binary
from scipy.io import loadmat
from tabulate import tabulate

import torchvision.transforms.functional as TF
import torchvision.transforms as T

import stacked_hourglass.res
from stacked_hourglass.datasets.common import DataInfo
from stacked_hourglass.utils.imutils import load_image, draw_labelmap
from stacked_hourglass.utils.misc import to_torch
from stacked_hourglass.utils.disk import getCache
from stacked_hourglass.utils.transforms import shufflelr, crop, color_normalize, fliplr, transform

class CSV(data.Dataset):

    # The ratio between input spatial resolution vs. output heatmap spatial resolution
    INPUT_OUTPUT_RATIO = 4

    def __init__(self, csv_path, data_folder_path,is_train=True, inp_res=256, sigma=1, scale_factor=0.25,
                 rot_factor=30, label_type='Gaussian',training_split=0.80):
        
        """
        Dataset which loads images from CSV file.
        
        Each row, starting on row two, should specify image/label entries
        The top two rows are reserved for column labels
        
        Each column should be for a separate label.
        
        All filepaths will be relative to data_folder_path input variable

        Args:
            csv_path: path to the CSV file
            
            data_folder_path: path to the root
            
            is_Train = if true, training dataset is loaded. If false, validation dataset entries are loaded
            
            inp_res (default = 256): height and width of input image
            
            sigma (default = 1): For keypoint labels, sigma of a guassian distribution centered on keypoint to construct image
            
            scale_factor (default = 0.25): AUGMENTATION VARIABLE. maximum scale factor during image augmentation
            
            rot_factor (default = 30): AUGMENTATION VARIABLE. maximum rotation factor (+/-) 
            
            label_type (default = 'Gaussian'): 
            
            training_split (default = 0.80): 
        """
        
        self.csv_path = csv_path
        self.data_folder = data_folder_path
        
        self.gray_mean = [0.1034]
        self.gray_std = [0.1425]
        
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
        
        self.label_list = []
        csvfile = open(self.csv_path,newline='')
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            self.label_list.append(row)
        
        #self.label_list = self.label_list[2:]
        
        indexes = [*range(0,len(self.label_list))]
        #random.shuffle(indexes) #The data should be shuffled before input.
        
        self.train_list = indexes[:int((len(indexes)+1)*training_split)] #Remaining 80% to training set
        self.valid_list = indexes[int((len(indexes)+1)*training_split):] 
        
        print('There are ', len(self.train_list), ' samples in the training dataset')
        print('There are ', len(self.valid_list), ' samples in the validation dataset')

    def __getitem__(self, index_input):
        
        sf = self.scale_factor
        rf = self.rot_factor
        
        if self.is_train:
            index = self.train_list[index_input]
        else:
            index = self.valid_list[index_input]
        
        #img loads image, converts to float32, and converts to a tensor
        img = getImage(self.label_list[index][0],self.data_folder,self.inp_res)  # CxHxW
        
        target = getImage(self.label_list[index][1],self.data_folder,self.out_res)
        
        if self.is_train:
            #Flip
            if random.random() <= 0.5:
                img = TF.hflip(img)
                target = TF.hflip(target)
            
            #Rotate
            if random.random() <= 0.5:
                r = random.uniform(-30.0,30.0)
                img = TF.rotate(img,r)
                target = TF.rotate(target,r)
                #target = TF.rotate(target,r,interpolation=T.InterpolationMode.BILINEAR) #This will preserve shape of Gaussian, but not make maximum 1
                
                #If we use bilinear for rotation, then the gaussian will probably no longer have a 
                #maximum of 1.0, so we can re-normalize.
                #max_values = target.reshape(target.shape[0],-1).max(dim=-1,keepdim=True)[0]
                #target = target / max_values.unsqueeze(2)
                #target[target.isnan()] = 0.0
                
            
            #Contrast
            if random.random() <= 0.5:
                img = TF.autocontrast(img)
                
            #Blur
            if random.random() <= 0.5:
                img = TF.gaussian_blur(img,3)
                
            #Invert
            if random.random() <= 0.5:
                img = TF.invert(img)
            # Scale
            #if random.random() <= 0.5:
            #    sf = torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0] #Scale between +/- scale factor
                
        # Prepare image and groundtruth map
        #inp = crop(img, c, s, self.inp_res, rot=r)
        inp = TF.normalize(img, self.gray_mean, self.gray_std)

        return inp, target

    def __len__(self):
        if self.is_train:
            return len(self.train_list)
        else:
            return len(self.valid_list)
        

#@raw_cache.memoize(typed=True)
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

#@raw_cache.memoize(typed=True)
def getImage(image_path,img_folder,inp_res):
    
    image = load_image(os.path.join(img_folder, image_path),'L')
    
    inp = TF.resize(image,inp_res,antialias=True)
    return inp

