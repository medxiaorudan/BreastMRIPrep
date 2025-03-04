import glob
import numpy as np
import SimpleITK as sitk
from torchvision import transforms, utils
from skimage import io, transform
import torch

class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    """
    
    def __init__(self, output_size):
        self.name='Rescale'
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        
    def __call__(self, sample):
        data, mask = sample['data'], sample['mask']
        z, y, x = data.shape
        new_z, new_y, new_x = self.output_size
        new_z, new_y, new_x = int(new_z), int(new_y), int(new_x)
        data = transform.resize(data, (new_z, new_y, new_x))
        mask = transform.resize(mask, (new_z, new_y, new_x), preserve_range=True)
        mask = np.where(mask>0,1,0)
        return {'data': data, 'mask': mask}

class ResampleImage(object):
    
    '''
    The code for resizing has been taken from
    https://gist.github.com/zivy/79d7ee0490faee1156c1277a78e4a4c4
    '''
    
    def __init__(self, output_size):
        self.name = 'resample image'
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        data, mask = sample['data'], sample['mask']
        new_z, new_y, new_x = self.output_size
        new_size = [new_x, new_y, new_z]
        interp_mask = sitk.sitkNearestNeighbor  # for labels
        interp_data = sitk.sitkLinear            # for input features
        data = sitk.GetImageFromArray(data)
        mask = sitk.GetImageFromArray(mask)
        reshaped_data = self.my_resample(data, new_size, interp_data)
        reshaped_mask = self.my_resample(mask, new_size, interp_mask)
        reshaped_data = sitk.GetArrayFromImage(reshaped_data)
        reshaped_mask = sitk.GetArrayFromImage(reshaped_mask)
        return {'data': reshaped_data, 'mask': reshaped_mask}
        
    def my_resample(self, img, new_size, interpolator):
        dimension = img.GetDimension()

        # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
        reference_physical_size = np.zeros(dimension)

        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    
        # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
        reference_origin = np.zeros(dimension)
        reference_direction = np.identity(dimension).flatten()
        reference_size = new_size
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

        reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
        # this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
        # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
        # spacing will not yield the correct coordinates resulting in a long debugging session. 
        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)
        # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth 
        # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that 
        # no new labels are introduced.
    
        return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)

    
class Normalization(object):
    """
    Normalize an image by setting its mean to zero and variance to one
    """

    def __init__(self):
        self.name = 'Normalization'

    def __call__(self, sample):
        rescFilter = sitk.RescaleIntensityImageFilter()
        rescFilter.SetOutputMaximum(255)
        rescFilter.SetOutputMinimum(0)
        data, mask = sample['data'], sample['mask']
        data = sitk.GetImageFromArray(data)
        data = rescFilter.Execute(data)
        data = sitk.GetArrayFromImage(data)
        return {'data':data, 'mask':mask}

class RandomFlip(object):
    """
    Randomly flipping volumes across all three axis
    """

    def __init__(self):
        self.name = 'RandomFlip'

    def __call__(self, sample):
    
        data, mask = sample['data'], sample['mask']
    
        # generating axis randomly
        flipaxes = np.random.random(3)>0.5
    
        flipdata = sitk.Flip(data, flipaxes.tolist())
        flipmask = sitk.Flip(mask, flipaxes.tolist())
        
        return {'data':flipdata, 'mask':flipmask}

class RandomSmoothing(object):
    """
    Randoml Gaussian smoothing
    """

    def __init__(self, prob):
        self.name = 'RandomSmoothing'
        self.prob = prob

    def __call__(self, sample):
        data, mask = sample['data'], sample['mask']
        if np.random.rand() < self.prob:
            data = sitk.GetImageFromArray(data)
            data = sitk.RecursiveGaussian(data)
            data = sitk.GetArrayFromImage(data)
        return {'data':data, 'mask':mask}

class RandomNoise(object):
    """
    Randomly Gaussian Noise 
    """
    
    def __init__(self, prob):
        self.name = 'RandomNoise'
        self.prob = prob

    def __call__(self, sample):
    
        data, mask = sample['data'], sample['mask']
    
        if np.random.rand() < self.prob:
            data = sitk.GetImageFromArray(data)
            data = sitk.AdditiveGaussianNoise(data)
            data = sitk.GetArrayFromImage(data)

        return {'data':data, 'mask':mask}


class HistogramMatching(object):
    """
    Histogram Matching with random images from training set (not applied to labels)
    """

    def __init__(self, data_dir, data_flag, train_size=40, prob=0.5):
        self.name = 'Histogram Match'
        self.train_size = train_size
        self.data_dir = data_dir
        self.prob = prob
        if data_flag == 'promise12':
            self.data_flag = 0
        elif data_flag == 'fj':
            self.data_flag = 1

    def __call__(self, sample):

        data, mask = sample['data'], sample['mask']
    
        # histogram matching with random image from training set
       
        if np.random.random() <= self.prob:
            if self.data_flag == 0:
                files = glob.glob(self.data_dir+'/Case*_segmentation.mhd')
                index = np.random.randint(0, self.train_size-1)
                file_name = files[index].replace('_segmentation', '')
            elif self.data_flag == 1:
                files = glob.glob(self.data_dir+'/pt*_t2.nii')
                index = np.random.randint(0, self.train_size-1)
                file_name = files[index]

            template = sitk.ReadImage(file_name)
            template = sitk.GetArrayFromImage(template)
            #source = sitk.GetArrayFromImage(data) // data already as ndarray 
            source = data
            oldshape = source.shape

            s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                  return_counts=True)
            t_values, t_counts = np.unique(template, return_counts=True)

            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]
            t_quantiles = np.cumsum(t_counts).astype(np.float64)
            t_quantiles /= t_quantiles[-1]

            interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

            data = interp_t_values[bin_idx].reshape(oldshape)
            #data = sitk.GetImageFromArray(data) # the data should be in ndarray format!

        return {'data':data, 'mask':mask}

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __init__(self, output_size):
        self.output_size = output_size
        self.z, self.y, self.x = output_size


    def __call__(self, sample):
        data, mask = sample['data'], sample['mask']
        data = torch.from_numpy(data).float().view(-1,self.z,self.y,self.x)
        mask = torch.from_numpy(mask).float().view(-1,self.z,self.y,self.x)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'data': data,
                'mask': mask}