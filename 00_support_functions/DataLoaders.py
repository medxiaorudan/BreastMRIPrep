# imports
import torch
from torch.utils.data import Dataset, DataLoader
import os, sys
import nibabel as nib

# loading DataGenerator to (potentially) augment the data
from DataGenerator import *


class TrainingDataset(Dataset):  
    """
    Dataloader for training dataset.

    Assumes each 
    """
    
    def __init__(self, directory, root_dir='.', read_size=(20,256,256), transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.path = directory
        
        self.data_extension = '*ref.nii'
        self.mask_extension = 'Segmentation.nii'

        self.data_file_list = []
        self.mask_file_list = []

        for file in glob.glob( os.path.join(directory, self.data_extension), recursive=True):
            self.data_file_list.append(file)
        for file in glob.glob( os.path.join(directory, self.mask_extension), recursive=True ):
            self.mask_file_list.append(file)

        self.data_file_list = sorted(self.data_file_list)
        self.mask_file_list = sorted(self.mask_file_list)

        # debugging purposes
        #for i in range( len(self.data_file_list)):
        #    print(list(zip(self.data_file_list, self.mask_file_list))[i])
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data_name = self.data_file_list[idx]
        mask_name = self.mask_file_list[idx]
        data = self.__load_nii__(data_name, False)
        mask = self.__load_nii__(mask_name, True)
        sample = {'data': data, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
            
        return(sample)
    
    def __len__(self):
        return( len(self.data_file_list) )
    
    def __load_nii__(self, filename, is_mask=False):
        nii_file = nib.load(filename)
        nii_data = nii_file.get_fdata() 
        nii_data = nii_data.swapaxes(0, 2)
        if is_mask:
            out = nii_data.astype(np.uint8)
        else:
            out = nii_data
        return out
    

class TrainingDataset2(Dataset):  
    """
    Dataloader for training dataset.

    Assumes each 
    """
    
    def __init__(self, directory, root_dir='.', transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        self.path = directory
        
        self.data_extension = '*ref.nii'
        self.mask_extension = 'Segmentation.nii'

        self.data_file_list = []
        self.mask_file_list = []

        for file in glob.glob( os.path.join(directory, self.data_extension), recursive=True):
            self.data_file_list.append(file)
        for file in glob.glob( os.path.join(directory, self.mask_extension), recursive=True ):
            self.mask_file_list.append(file)

        self.data_file_list = sorted(self.data_file_list)
        self.mask_file_list = sorted(self.mask_file_list)

        # debugging purposes
        #for i in range( len(self.data_file_list)):
        #    print(list(zip(self.data_file_list, self.mask_file_list))[i])
        
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()  
        
        if (idx >= len(self.data_file_list)):
            idx = idx % len(self.data_file_list) 

        data_name = self.data_file_list[idx]
        mask_name = self.mask_file_list[idx]
        
        if idx < len(self.data_file_list):
            # get left breast
            data = self.__load_nii__(data_name, "left")
            mask = self.__load_nii__(mask_name, "left", True)
            sample = {'data': data, 'mask': mask}
        elif idx >= len(self.data_file_list) and idx < len(self.data_file_list)*2:
            # get right breast, and mirror it
            data = self.__load_nii__(data_name, "right")
            mask = self.__load_nii__(mask_name, "right", True)
        else:
            print("Error! Index out of bounds!")
            return( None )
        
        if self.transform:
            sample = self.transform(sample)
            
        return(sample)
    
    def __len__(self):
        return( len(self.data_file_list)*2 )

    def __load_nii__(self, filename, left_or_right="left", is_mask=False):
        nii_file = nib.load(filename)
        nii_data = nii_file.get_fdata() 
        nii_data = nii_data.swapaxes(0, 2)

        nc = nii_data.shape[2]
        mp = int(np.floor(nc/2))
        if left_or_right == "left":
            return( nii_data[:, :, 0:mp] )
        elif left_or_right == "right":
            return( np.flip(nii_data[:, :, mp:nc], 2) )
        else:
            print("Error! Invalid left/right laterality!")
            return( None )

