 
 
from ast import Raise
import glob
import random
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from utils import Custom_Compose, min_max_scaling



class ImageDataset(Dataset):
    def __init__(self, inputs, outputs, metadata, 
                 i_seq, o_seq, roi_size = 32, #path,
                 batch_size=10, image_size = (512,512), 
                 n_channels = 1,
                 shuffle = True,
                 transforms_=None, name="dce-mri", num_workers=1, norm = "z-score", **kwargs):
        #
        'Initialization'
        self.files = inputs  #np.array(inputs)  # list_IDs
        self.targs = outputs #np.array(outputs)   # labels
        self.dim = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.roi_size = roi_size
        self.i_seq = i_seq
        self.o_seq = o_seq
        self.name = name
        self.num_workers = num_workers 
        # self.norm = norm

        self.metadata = metadata

        if norm != "ti_norm": 
            self.transforms =  T.Compose(transforms_) if isinstance(transforms_, list) else transforms_ 
        else: 
            self.transforms =  Custom_Compose(transforms_) if isinstance(transforms_, list) else transforms_ 
    
    
    def transform_roi_points (self, p, p_, matrix): 
        #
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        p_a = (int(px), int(py))
            
        px = (matrix[0][0]*p_[0] + matrix[0][1]*p_[1] + matrix[0][2]) / ((matrix[2][0]*p_[0] + matrix[2][1]*p_[1] + matrix[2][2]))
        py = (matrix[1][0]*p_[0] + matrix[1][1]*p_[1] + matrix[1][2]) / ((matrix[2][0]*p_[0] + matrix[2][1]*p_[1] + matrix[2][2]))
        p__a = (int(px), int(py))
        
        return p_a, p__a
    

    def __random_shuffle__(self): 
        #
        self.files, self.targs, self.metadata = shuffle(self.files, self.targs, self.metadata )
        self.metadata.index = np.arange(len(self.files)) 


    def __getitem__(self, index):
        #
        if isinstance( index, int ) :
            sample = index % len(self.files)
            if self.name == "dce-mri":
                im_input_  = Image.open(self.files[sample]).convert("F") # Check (1 channel)
                im_output_ = Image.open(self.targs[sample]).convert("F")
                names = self.metadata["patient"].iloc[sample] + "_" + self.metadata["ROI"].iloc[sample]

            if self.name == "duke":
                im_input_  = Image.open(self.files[sample]).convert("F")
                im_output_ = Image.open(self.targs[sample]).convert("F")
                names = self.metadata["Patient ID"].iloc[sample]
            
            if isinstance(self.transforms, Custom_Compose):
                #
                m, s = self.metadata["mean_"].iloc[sample], self.metadata["std_"].iloc[sample]
                im_input_  = self.transforms(im_input_,  m, s)
                im_output_ = self.transforms(im_output_, m, s)
            else:
                #
                im_input_  = self.transforms(im_input_)
                im_output_ = self.transforms(im_output_)
            
            return {"in": im_input_, "out": im_output_, "ids": names}
        
        elif isinstance( index, slice ) :
            batch_size_ = np.abs(index.indices(len(self))[0] - index.indices(len(self))[1])
            batch_input_  = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            batch_output_ = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            names = []
            for i, idx in enumerate(range(*index.indices(len(self)))): 
                sample = idx % len(self.files)
                
                if self.name == "dce-mri":
                    im_input_  = Image.open(self.files[sample]).convert("F")
                    im_output_ = Image.open(self.targs[sample]).convert("F")
                    names.append (self.metadata["patient"].iloc[sample] + "_" + self.metadata["ROI"].iloc[sample])
                if self.name == "duke":
                    im_input_  = Image.open(self.files[sample]).convert("F")
                    im_output_ = Image.open(self.targs[sample]).convert("F")
                    names.append (self.metadata["Patient ID"].iloc[sample])
                    if np.array(im_output_).min()==0 and np.array(im_output_).max()==0:
                        im_input_  = Image.open(self.files[sample-1]).convert("F")
                        im_output_ = Image.open(self.targs[sample-1]).convert("F")
                
                if isinstance(self.transforms, Custom_Compose):
                    #
                    m, s = self.metadata["mean_"].iloc[sample], self.metadata["std_"].iloc[sample]
                    im_input_  = self.transforms(im_input_,  m, s )
                    im_output_ = self.transforms(im_output_, m, s )
                else: 
                    im_input_  = self.transforms(im_input_)
                    im_output_ = self.transforms(im_output_)

                batch_input_[i] = im_input_; batch_output_[i] = im_output_
                
            return {"in": batch_input_, "out": batch_output_, "ids": names}

    def __len__(self):
        return len(self.files)


class ValImageDataset(Dataset):
    def __init__(self, inputs, outputs, metadata, 
                 i_seq, o_seq, roi_size = (64,64), #path,
                 batch_size=10, 
                 image_size = (512,512), 
                 n_channels = 1,
                 shuffle = True, norm = "z-score",
                 transforms_ = None, roi_transform = None, name="dce-mri", **kwargs):
        #
        'Initialization'
        self.files = inputs  #np.array(inputs)  # list_IDs
        self.targs = outputs #np.array(outputs)   # labels
        self.dim = image_size
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.roi_size = roi_size
        self.i_seq = i_seq
        self.o_seq = o_seq
        self.name = name

        self.metadata = metadata

        if norm != "ti_norm": 
            self.transforms = T.Compose(transforms_) if isinstance(transforms_, list) else transforms_
            self.roi_transform = T.Compose(roi_transform) if isinstance(roi_transform, list) else roi_transform
        else: 
            self.transforms = Custom_Compose(transforms_) if isinstance(transforms_, list) else transforms_
            self.roi_transform = Custom_Compose(roi_transform) if isinstance(roi_transform, list) else roi_transform

    
    def transform_roi_points (self, p, p_, matrix): 
        #
        px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
        p_a = (int(px), int(py))
            
        px = (matrix[0][0]*p_[0] + matrix[0][1]*p_[1] + matrix[0][2]) / ((matrix[2][0]*p_[0] + matrix[2][1]*p_[1] + matrix[2][2]))
        py = (matrix[1][0]*p_[0] + matrix[1][1]*p_[1] + matrix[1][2]) / ((matrix[2][0]*p_[0] + matrix[2][1]*p_[1] + matrix[2][2]))
        p__a = (int(px), int(py))
        
        return p_a, p__a
    

    def __random_shuffle__(self): 
        #
        self.files, self.targs, self.metadata = shuffle(self.files, self.targs, self.metadata )
        self.metadata.index = np.arange(len(self.files)) 
    

    def extract_roi(self, im, idx):
        #
        if self.name == "dce-mri":
            i_roi_size = [round(float(self.metadata["Distancia_x_" + self.i_seq][idx])), round(float(self.metadata["Distancia_y_" + self.i_seq][idx])), 1]
            point = (round(float(self.metadata["Centro_x_" + self.i_seq][idx])), round(float(self.metadata["Centro_y_" + self.i_seq][idx])))
            L_xi = point[0] - int((i_roi_size[0]) / 2); L_xi = L_xi-1
            L_yi = point[1] - int((i_roi_size[1]) / 2); L_yi = L_yi-1
        
        elif self.name == "duke": 
            # Start_Row	End_Row	Start_Column	End_Column
            row_s = abs(self.metadata["End_Row"][idx] - self.metadata["Start_Row"][idx])
            col_s = abs(self.metadata["End_Column"][idx] - self.metadata["Start_Column"][idx])
            i_roi_size = [row_s, col_s, 1]
            L_xi = min(self.metadata["Start_Column"][idx], self.metadata["End_Column"][idx])
            L_yi = min(self.metadata["Start_Row"][idx], self.metadata["End_Row"][idx])
        
        if im.shape[0] == self.n_channels:
            roi = im[:, L_yi:(L_yi+i_roi_size[0]), L_xi:L_xi+i_roi_size[1]]
        elif len(im.shape) > 2: 
            if im.shape[2] == self.n_channels:
                roi = im[L_yi:(L_yi+i_roi_size[0]), L_xi:L_xi+i_roi_size[1], :]
        else: 
            roi = im[L_yi:(L_yi+i_roi_size[0]), L_xi:L_xi+i_roi_size[1]]
        
        pts_1 = np.float32([[0,0],[im.shape[0],0],[0,im.shape[1]],[im.shape[0],im.shape[1]]]) 
        pts_2 = np.float32([[0,0],[self.dim[0],0],[0,self.dim[1]],[self.dim[0],self.dim[1]]]) 
        matrix_p = cv2.getPerspectiveTransform(pts_1, pts_2)
        
        ps, pe = (L_xi, L_yi), (L_xi+i_roi_size[1], L_yi+i_roi_size[0])
        psT, peT = self.transform_roi_points(ps, pe, matrix_p)

        return roi, (psT, peT)


    def __getitem__(self, index):
        #
        if isinstance( index, int ) :
            sample = index % len(self.files)
            
            if self.name == "dce-mri":
                im_input_  = Image.open(self.files[sample]).convert("F")
                im_output_ = Image.open(self.targs[sample]).convert("F")
                names = self.metadata["patient"].iloc[sample] + "_" + self.metadata["ROI"].iloc[sample]
            if self.name == "duke":
                im_input_  = Image.open(self.files[sample]).convert("F")
                im_output_ = Image.open(self.targs[sample]).convert("F")
                names = self.metadata["Patient ID"].iloc[sample]
            
            roi_input_,  pT_input_  = self.extract_roi( np.array(im_input_), index)
            roi_output_, pT_output_ = self.extract_roi(np.array(im_output_), index)
            
            # # m, s = np.mean(im_input_), np.std(im_input_)
            m, s = self.metadata["mean_"].iloc[sample], self.metadata["std_"].iloc[sample]

            if isinstance(self.transforms, Custom_Compose):
                #
                im_input_  = self.transforms(im_input_,  m, s)
                im_output_ = self.transforms(im_output_, m, s)
                roi_input_  = self.roi_transform(Image.fromarray(roi_input_),  m, s)
                roi_output_ = self.roi_transform(Image.fromarray(roi_output_), m, s)
            else:
                #
                im_input_  = self.transforms(im_input_)
                im_output_ = self.transforms(im_output_)
                roi_input_  = self.roi_transform(Image.fromarray(roi_input_))
                roi_output_ = self.roi_transform(Image.fromarray(roi_output_))
            
            return {"in": im_input_, "out": im_output_, 
                    "roi_in": roi_input_, "roi_out": roi_output_, 
                    "pt_roi_in": pT_input_, "pt_roi_out": pT_output_,
                    "ids": names }

        elif isinstance( index, slice ) :
            batch_size_ = np.abs(index.indices(len(self))[0] - index.indices(len(self))[1])
            batch_input_  = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            batch_output_ = torch.empty([batch_size_, self.n_channels, *self.dim]); 
            batch_roi_input_  = torch.empty([batch_size_, self.n_channels, *self.roi_size]); 
            batch_roi_output_ = torch.empty([batch_size_, self.n_channels, *self.roi_size]); 
            batch_pt_input_, batch_pt_output_, names = [], [], [] 
            
            for i, idx in enumerate(range(*index.indices(len(self)))): 
                sample = idx % len(self.files)
                
                if self.name == "dce-mri":
                    im_input_  = Image.open(self.files[sample]).convert("L")
                    im_output_ = Image.open(self.targs[sample]).convert("L")
                    names.append (self.metadata["patient"].iloc[sample] + "_" + self.metadata["ROI"].iloc[sample])
                if self.name == "duke":
                    im_input_  = Image.open(self.files[sample]).convert("F")
                    im_output_ = Image.open(self.targs[sample]).convert("F")
                    names.append (self.metadata["Patient ID"].iloc[sample])
                    
                roi_input_,  pT_input_  = self.extract_roi( np.array(im_input_), idx)
                roi_output_, pT_output_ = self.extract_roi(np.array(im_output_), idx)
                
                if isinstance(self.transforms, Custom_Compose):
                    #
                    m, s = self.metadata["mean_"].iloc[sample], self.metadata["std_"].iloc[sample]
                    im_input_  = self.transforms(im_input_,  m, s )
                    im_output_ = self.transforms(im_output_, m, s )
                    roi_input_  = self.roi_transform(Image.fromarray(np.array(roi_input_)),  m, s )
                    roi_output_ = self.roi_transform(Image.fromarray(np.array(roi_output_)), m, s )
                else:
                    #
                    im_input_  = self.transforms(im_input_)
                    im_output_ = self.transforms(im_output_)
                    roi_input_  = self.roi_transform(Image.fromarray(np.array(roi_input_)))
                    roi_output_ = self.roi_transform(Image.fromarray(np.array(roi_output_)))
                
                batch_input_[i] = im_input_; batch_output_[i] = im_output_
                batch_roi_input_[i] = roi_input_; batch_roi_output_[i] = roi_output_
                batch_pt_input_.append(pT_input_), batch_pt_output_.append(pT_output_)
                
            return {"in": batch_input_, "out": batch_output_, 
                    "roi_in": batch_roi_input_, "roi_out": batch_roi_output_, 
                    "pt_roi_in": batch_pt_input_, "pt_roi_out": batch_pt_output_,
                    "ids": names}

    def __len__(self):
        return len(self.files)


class Loader():
    def __init__(self, input_sequence, output_sequence, data_path,
                 batch_size=50, roi_size = (64,64), dataset_name = "dce-mri", quality = None,
                 train_transforms = None, val_transforms = None, roi_transforms = None,
                 img_res=(128, 128), n_channels = 3, workers = 0, norm = "z-score", **kwargs):
        #
        self.data_path = data_path
        
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence
        self.batch_size = batch_size
        self.roi_size = roi_size
        self.dataset_name = dataset_name.lower()
        self.quality = quality
        self.img_res = img_res
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.roi_transforms = roi_transforms
        self.n_channels = n_channels
        self.num_workers=workers
        self.norm = norm
        self.img_res = img_res
        self.roi_size = roi_size

        allowed_datasets = ["dce-mri", "duke"]
        assert dataset_name.lower() in allowed_datasets, NotImplementedError ("{0}: Dataset not implemented".format(dataset_name))

        if dataset_name.lower() == "dce-mri":
            allowed_sequences = ["t1","t2","adc","dif","d0","d1","d2",
                                "d3","d4","d5","f1","f2","f3","f4","f5"]
            diff_adc = ["adc","dif"]
        
        elif dataset_name.lower() == "duke":
            allowed_sequences = ["pre","post_1","post_2","post_3","post_4"]
            assert quality in ["1T", "3T"], ValueError ("What's the quality of the images?! Valid only 1T, 3T")
        
        
        if input_sequence not in allowed_sequences: 
            raise ValueError("Error! Check input_sequence value. Current: {0}".format(str(input_sequence)))
        
        if output_sequence not in allowed_sequences: 
            raise ValueError("Error! Check output_sequence value. Current: {0}".format(str(output_sequence)))
        
        
        if dataset_name.lower() == "dce-mri":
            (train_i, train_o, train_meta), (test_i, test_o, test_meta) = self.get_dcemri_metadata(diff_adc)
            # (train_i, train_o, train_meta), (test_i, test_o, test_meta), (val_i, val_o, val_meta) = self.get_dcemri_metadata(diff_adc)

        elif dataset_name.lower() == "duke":
            (train_i, train_o, train_meta), (test_i, test_o, test_meta) = self.get_duke_metadata()
            # (train_i, train_o, train_meta), (test_i, test_o, test_meta), (val_i, val_o, val_meta) = self.get_duke_metadata()
        
        else: 
            raise NotImplementedError (dataset_name, "Database not implemented")
        

        self.train_generator = ValImageDataset(train_i, train_o, train_meta, name=self.dataset_name, #ImageDataset
                                            i_seq = input_sequence, o_seq = output_sequence,
                                            roi_size = roi_size, batch_size=batch_size,
                                            image_size=img_res, n_channels=n_channels, norm = self.norm,
                                            shuffle = False, transforms_ = self.train_transforms, roi_transform = self.roi_transforms, num_workers=workers)
        

        self.test_generator = ValImageDataset ( test_i, test_o, test_meta, name=self.dataset_name,
                                                i_seq = input_sequence, o_seq = output_sequence,
                                                roi_size = roi_size, batch_size=batch_size,
                                                image_size=img_res, n_channels=n_channels, norm = self.norm,
                                                transforms_ = self.val_transforms, roi_transform = self.roi_transforms, 
                                                shuffle = False, num_workers=workers)
        
        ## "Efficient" data loader
        self.train_loader = DataLoader(dataset     = self.train_generator, # use custom created train Dataset
                                       batch_size  = batch_size, # how many samples per batch?
                                       num_workers = workers, # how many subprocesses to use for data loading? (higher = more)
                                       shuffle     = False ) # shuffle the data?
        
        self.test_loader = DataLoader(dataset     = self.test_generator, # use custom created train Dataset
                                      batch_size  = batch_size, # how many samples per batch?
                                      num_workers = workers, # how many subprocesses to use for data loading? (higher = more)
                                      shuffle     = False ) # shuffle the data?

    
    def get_dcemri_metadata(self, diff_adc):#
        #
        if self.input_sequence in diff_adc or self.output_sequence in diff_adc :
            train = self.data_path + "train_ad.csv"
            test = self.data_path + "test_ad.csv"
            # val = self.data_path + "val_ad.csv"
        else:
            train = self.data_path + "train_c.csv"
            test = self.data_path + "test_c.csv"
            # val = self.data_path + "val_c.csv"
        
        train = pd.read_csv(train)
        test = pd.read_csv(test)
        # val = pd.read_csv(val)

        train_i = [self.data_path+self.input_sequence+"/train/"+img for img in train["A2_"+self.input_sequence]] 
        train_o = [self.data_path+self.output_sequence+"/train/"+img for img in train["A2_"+self.output_sequence]]
        
        test_i = [self.data_path+self.input_sequence+"/test/"+img for img in test["A2_"+self.input_sequence]] 
        test_o = [self.data_path+self.output_sequence+"/test/"+img for img in test["A2_"+self.output_sequence]]
        
        # val_i = [self.data_path+self.input_sequence+"/val/"+img for img in val["A2_"+self.input_sequence]] 
        # val_o = [self.data_path+self.output_sequence+"/val/"+img for img in val["A2_"+self.output_sequence]]

        train = train.filter(["patient", "ROI", "birads", "Distancia_x_" + self.input_sequence, "Distancia_y_" + self.input_sequence, "Centro_x_" + self.input_sequence, "Centro_y_" + self.input_sequence, "Centro_z_" + self.input_sequence, "Distancia_x_" + self.output_sequence, "Distancia_y_" + self.output_sequence, "Centro_x_" + self.output_sequence, "Centro_y_" + self.output_sequence, "Centro_z_" + self.output_sequence, "mean_", "std_"])
        test = test.filter(["patient", "ROI", "birads", "Distancia_x_" + self.input_sequence, "Distancia_y_" + self.input_sequence, "Centro_x_" + self.input_sequence, "Centro_y_" + self.input_sequence, "Centro_z_" + self.input_sequence, "Distancia_x_" + self.output_sequence, "Distancia_y_" + self.output_sequence, "Centro_x_" + self.output_sequence, "Centro_y_" + self.output_sequence, "Centro_z_" + self.output_sequence, "mean_", "std_"])
        # val = val.filter(["patient", "ROI", "birads", "Distancia_x_" + self.input_sequence, "Distancia_y_" + self.input_sequence, "Centro_x_" + self.input_sequence, "Centro_y_" + self.input_sequence, "Centro_z_" + self.input_sequence, "Distancia_x_" + self.output_sequence, "Distancia_y_" + self.output_sequence, "Centro_x_" + self.output_sequence, "Centro_y_" + self.output_sequence, "Centro_z_" + self.output_sequence, "mean_d0", "std_d0"])
        
        return (train_i, train_o, train), (test_i, test_o, test) #, (val_i, val_o, val)

    
    def get_duke_metadata(self):#
        #
        train = self.data_path + "train_{0}.csv".format(self.quality)
        test = self.data_path + "test_{0}.csv".format(self.quality)
        # val = self.data_path + "val_{0}.csv".format(self.quality)

        train = pd.read_csv(train)
        test = pd.read_csv(test)
        # val = pd.read_csv(val)
        
        train_i = [self.data_path+img for img in train[self.input_sequence]] 
        train_o = [self.data_path+img for img in train[self.output_sequence]]
        
        test_i = [self.data_path+img for img in test[self.input_sequence]] 
        test_o = [self.data_path+img for img in test[self.output_sequence]]
        
        # val_i = [self.data_path+img for img in val[self.input_sequence]] 
        # val_o = [self.data_path+img for img in val[self.output_sequence]]

        train = train.filter(["Patient ID", "Slice", "Start_Row", "End_Row", "Start_Column", "End_Column", "magnetic_field", "ROI deep", self.input_sequence, self.output_sequence, "mean_", "std_"])
        test = test.filter(["Patient ID", "Slice", "Start_Row", "End_Row", "Start_Column", "End_Column", "magnetic_field", "ROI deep", self.input_sequence, self.output_sequence, "mean_", "std_"])
        # val = val.filter(["Patient ID", "Slice", "Start_Row", "End_Row", "Start_Column", "End_Column", "magnetic_field", "ROI deep", self.input_sequence, self.output_sequence])
        
        return (train_i, train_o, train), (test_i, test_o, test) #, (val_i, val_o, val)


    def on_epoch_end(self, shuffle = "train"): 
        #
        assert shuffle.lower() in ["train", "test", "val"], "Check subset to shuffle"

        if shuffle.lower() == "train": 
            self.train_generator.__random_shuffle__()
        elif shuffle.lower() == "test": 
            self.test_generator.__random_shuffle__()
        elif shuffle.lower() == "val": 
            self.val_generator.__random_shuffle__()
    

    def __len__(self):
        return len(self.train_generator)



"""

# Configure dataloaders
transforms = [
    T.Resize((256, 256), Image.BICUBIC),
    T.ToTensor(),
    min_max_scaling(out_range = [-1,1])
]


# Configure data loader
dataset_name = 'DUKE'
data_loader = Loader(input_sequence = "post_1", output_sequence = "post_3", dataset_name = dataset_name,
                     data_path="/media/ruben-kubuntu/Datos/duke_data/Duke_tiff/", quality = "1T",
                     batch_size = 5, img_res=(256, 256), roi_size = (64,64), n_channels = 1, 
                     transforms = transforms)


images = data_loader.train_generator[20:30]
print (images["in"].shape, images["out"].shape)


import matplotlib.pyplot as plt
_, axes = plt.subplots(2,2,figsize=(10,10))
axes = axes.ravel()
axes[0].imshow(np.squeeze(images["in"][5].detach().numpy()), cmap="gray")
axes[1].imshow(np.squeeze(images["out"][5].detach().numpy()), cmap="gray")
# axes[2].imshow(np.squeeze(images["roi_in"][5].detach().numpy()), cmap="gray")
# axes[3].imshow(np.squeeze(images["roi_out"][5].detach().numpy()), cmap="gray")
plt.savefig("im1.png"); plt.close()
"""