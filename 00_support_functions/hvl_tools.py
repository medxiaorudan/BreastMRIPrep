#functions I use often
import numpy as np
import matplotlib.pyplot as plt
import torch
#import SimpleITK as sitk
from ipywidgets import interact,fixed
import ipywidgets as widgets
import matplotlib.patches as patches
import imageio
import pydicom as dicom
import os

#makes an animated image out of 3d numpy arrays
def make_gif(arr,filename):
    sarr=np.squeeze(arr)
    if len(sarr.shape)!=3:
        print('Invalid array size!')
        return
    else:
        sarr=sarr/sarr.max()*255
        imageio.mimwrite(filename, sarr.astype('uint8'),fps=30)

#simplistic plot of pytorch tensor
def my_plot(x,y,npl=True):
    if type(x) !=  np.ndarray:  x=np.squeeze(x.detach().cpu().numpy())
    if type(y) !=  np.ndarray:  y=np.squeeze(y.detach().cpu().numpy())
    if npl: plt.figure(figsize=(15,10))
    plt.plot(x,y,'.-')

#convert 1d pytorch tensor into a matplotlib plot, option for fft display
def t_plot(x,y,npl=True,dyn_scan_time=1,fft=True):
    if type(x) !=  np.ndarray:  x=np.squeeze(x.detach().cpu().numpy())
    if type(y) !=  np.ndarray:  y=np.squeeze(y.detach().cpu().numpy())
    if npl: plt.figure(figsize=(15,10))
    plt.subplot(211)
    plt.plot(x,y,'.-')
    plt.title('time course')
    plt.xlabel('Time (s)')
    if fft:
        plt.subplot(212)
        N=y.shape[-1]
        w=np.fft.fftfreq(N,dyn_scan_time)[:N//2]
        fft=np.fft.fft(y)[:N//2]*2/N
        plt.plot(w[1:],np.abs(fft)[1:],'-')
        plt.title('FFT')
        plt.xlabel('Freq (Hz)')

#interactive display of 3d images, takes pytorch, sitk and np arrays
def imyshow(im):
    def myshow(arr,vmin,vmax,z=0,t=0):
        plt.imshow(arr[t,z],vmin=vmin,vmax=vmax,cmap='gray')  
    if type(im)==torch.Tensor:
        arr=im.cpu().detach().numpy()
    elif type(im)==sitk.SimpleITK.Image:
        arr=sitk.GetArrayFromImage(im)
    elif type(im)==np.ndarray or type(im)==np.core.memmap:
        arr=np.copy(im)
    else:
        print('Unknown format, nothing to display!')
        return None
    if len(arr.shape)>3:
        arr=np.squeeze(arr)
    print(arr.shape)
    if len(arr.shape)==2:
        interact(myshow,arr=fixed(arr[None,None,:]),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=fixed(0),t=fixed(0))
    if len(arr.shape)==3:
        interact(myshow,arr=fixed(arr[None,:]),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=(1,arr.shape[0]-1),t=fixed(0))
    if len(arr.shape)==4:
        interact(myshow,arr=fixed(arr),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=(1,arr.shape[1]-1),t=(0,arr.shape[0]-1))

     
# expanded interactive display of 3d images, takes pytorch, sitk and np arrays
def imyshowWOL(im, ol, op=0.1):
    def myshow(arr,ol,vmin,vmax,z=0,t=0):
        plt.imshow(arr[t,z],vmin=vmin,vmax=vmax,cmap='gray')  
        plt.imshow(ol[t,z],vmin=vmin,vmax=vmax,cmap='hot',alpha=op)
    if type(im)==torch.Tensor:
        arr=im.cpu().detach().numpy()
        ol=ol.cpu().detach().numpy()
    elif type(im)==sitk.SimpleITK.Image:
        arr=sitk.GetArrayFromImage(im)
        ol=sitk.GetArrayFromImage(ol)
    elif type(im)==np.ndarray or type(im)==np.core.memmap:
        arr=np.copy(im)
        ol=np.copy(ol)
    else:
        print('Unknown format, nothing to display!')
        return None
    if len(arr.shape)>3:
        arr=np.squeeze(arr)
        ol=np.squeeze(ol)
    print(arr.shape)
    if len(arr.shape)==2:
        interact(myshow,arr=fixed(arr[None,None,:]),ol=fixed(ol[None,None,:]),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=fixed(0),t=fixed(0))
    if len(arr.shape)==3:
        interact(myshow,arr=fixed(arr[None,:]),ol=fixed(ol[None,:]),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=(1,arr.shape[0]-1),t=fixed(0))
    if len(arr.shape)==4:
        interact(myshow,arr=fixed(arr),ol=fixed(ol),vmin=fixed(arr.min()),vmax=fixed(arr.max()),z=(1,arr.shape[1]-1),t=(0,arr.shape[0]-1))
        
#display time series, time course and corresponding fft inside a roi
def roi_show(arr,z=0,y=0,x=0,roi_size=1,dyn_scan_time=1):
    N=len(arr)
    plt.figure(figsize=(30,10))
    ax_im=plt.subplot(231)
    ax_im.imshow(arr[z],cmap='gray',vmin=arr.min(),vmax=arr.max())
    rect_ur = patches.Rectangle((x-.5,y-.5),roi_size,roi_size,linewidth=1,edgecolor='r',facecolor='none')
    ax_im.add_patch(rect_ur)
    ax_time=plt.subplot(232)
    mean_roi_sp_ur=np.mean(arr[:,y:y+roi_size,x:x+roi_size],axis=(1,2))
    t=np.linspace(0,dyn_scan_time*(N-1),N)    #np.arange(0,dyn_scan_time*N,dyn_scan_time) 
    ax_time.plot(t,mean_roi_sp_ur,'.-')
    ax_time.set_ylabel('Mean roi intensity')
    ax_time.set_xlabel('Time(s)')
    ax_time.scatter(z*dyn_scan_time,mean_roi_sp_ur[z],c='red')
    ax_ur_fft=plt.subplot(233)
    fft_sp_ur=np.fft.fft(mean_roi_sp_ur)[:N//2]*2/N 
    f_ur=np.fft.fftfreq(N,dyn_scan_time)[:N//2] 
    ax_ur_fft.plot(f_ur[1:],np.abs(fft_sp_ur[1:]))
    ax_ur_fft.set_ylabel('amplitude')
    ax_ur_fft.set_xlabel('Frequency (Hz)')

#interactive version to explore a time series
def iroi_show(arr,dyn_scan_time=1):    
    layout=widgets.Layout(width='90%')
    interact(roi_show,
             arr=fixed(arr),
             z=widgets.IntSlider(min=0,max=arr.shape[0]-1,step=1,value=0,layout=layout),
             y=widgets.IntSlider(min=0,max=arr.shape[1]-1,step=1,value=arr.shape[1]//2,layout=layout),
             x=widgets.IntSlider(min=0,max=arr.shape[2]-1,step=1,value=arr.shape[2]//2,layout=layout),
             roi_size=widgets.IntSlider(min=1,max=min(arr.shape[1],arr.shape[2]),step=1,value=1),
            dyn_scan_time=fixed(dyn_scan_time))
    
#Simpleelastix registration,b-spline only, applied on a 3d np array    
def apply_el_reg(arr,ref_index,nit='256'):                
    pmap=sitk.GetDefaultParameterMap('bspline')                                 #no affine tsf, all images should be aligned
    pmap['MaximumNumberOfIterations']=[nit]                                    #default is 256, lower to improve time
    pmap['FinalBSplineInterpolationOrder']=['3']
    pmap['MaximumNumberOfSamplingAttempts']=['8']
    pmap['Metric0Weight']=['0.9999']
    pmap['Metric1Weight']=['0.0001']
    pmap['FinalGridSpacingInPhysicalUnits']=['12']
    pmap['NumberOfResolutions']=['4']
    el_reg=sitk.ElastixImageFilter()
    el_reg.SetParameterMap(pmap)
    el_reg.SetLogToConsole(True)
    el_reg.SetLogToFile(True)
    el_reg.SetFixedImage(sitk.GetImageFromArray(arr[ref_index]))
    arr_list=[]
    def_list=[]
    moving_stack=sitk.GetImageFromArray(arr)
    print('Registering',moving_stack.GetDepth(),'images...')
    for i in range(moving_stack.GetDepth()):
        print(i+1,end=' , ')
        el_reg.SetMovingImage(moving_stack[:,:,i]) 
        el_reg.Execute()
        arr_list.append(sitk.GetArrayFromImage(el_reg.GetResultImage()))
        def_list.append(el_reg.GetTransformParameterMap())
    return np.array(arr_list),def_list

def read_folder_and_parse(inputfolder,tag_list,verbose=True): #first tag must be common identifier for an image stack, e.g. SeriesInstanceUID (0x20, 0x0e)
    class dataslice:                                              
        def __init__(self,file,tag_list):
            self.pixel_array=file.pixel_array
            for tag in tag_list:
                if tag==(0x08,0x08):
                    self.__dict__[tag]=file[tag].value[2]                        #typically R,I,P,M
                else:
                    self.__dict__[tag]=file[tag].value
    class output_object:
        def __init__(self,tag_vals,output_array,tag_names,refslice):
            self.output_array=output_array
            self.tag_vals=tag_vals            
            self.tag_names=tag_names
            self.refslice=refslice
    if verbose==True:
        print('\nMESSAGE: Parsing started') 
        print('\nMESSAGE: Reading files in input folder:\n',inputfolder,'\n')       
    tag_list=[dicom.tag.Tag(tag) for tag in tag_list]                               #convert to dicom.tag.Tag type
    if verbose==True: print('MESSAGE:Commencing parsing, sought after dicom tags are:\n',[tag for tag in tag_list],'\n')
    ds_list=[]
    excluded_files_list=[]
    missing_tag_files_list=[]
    refslices_list=[]
    outputs_list=[]
    UID_list=[]
    if not os.path.exists(inputfolder):                                            
        print('WARNING: Input folder does not exist!\n')
        return
    for dirName,  subdirList,  fileList in os.walk(inputfolder):
        for filename in fileList:
            try:
                file=dicom.read_file(os.path.join(dirName, filename)) 
                if 'PixelData' in file:                                                                           #check if image data is present in the file
                    try:
                        ds=dataslice(file,tag_list)
                        ds_list.append(ds)
                        if len(ds_list)==1:
                            tag_names=[file[tag].name for tag in tag_list]
                            if verbose==True: print('MESSAGE: Tag names are:\n',tag_names,'\n')
                        UID=ds.__dict__[tag_list[0]]
                        if UID not in UID_list:
                            UID_list.append(UID)
                            refslices_list.append(file)
                            outputs_list.append([ds])
                        else:
                            if ds.pixel_array.shape == outputs_list[UID_list.index(UID)][-1].pixel_array.shape:       #check that the dims are the same as previous slice
                                outputs_list[UID_list.index(UID)].append(ds)   #all slices belonging together in image stack i are now in the same list in outputs_list[i]
                            else:
                                raise NameError('WARNING: Found images with same UID but different dimensions')
                    except(KeyError):
                        missing_tag_files_list.append(filename)
            except(dicom.errors.InvalidDicomError):
                excluded_files_list.append(filename)
    if verbose==True:
        if excluded_files_list:
            print('WARNING: the following files are not valid dicoms files and have been excluded:\n',excluded_files_list,end='\n\n')
        if missing_tag_files_list:
            print('WARNING: the following dicom files do not have the sought after dicom tags and have been excluded:\n', missing_tag_files_list,'\n')
        print('MESSAGE: Found',len(ds_list),'valid files \n')
        print('MESSAGE: Found',len(UID_list), 'valid series \n')
    outputs=[]
    for output,UID in zip(outputs_list,UID_list):             #gather all possible tag values for images in same stack, in the order they appear in given tag_list
        tag_vals=[[] for i in range(len(tag_list)-1)]
        for ds in output:
            for i in range(len(tag_list)-1):
                if ds.__dict__[tag_list[i+1]] not in tag_vals[i]:
                    tag_vals[i].append(ds.__dict__[tag_list[i+1]])
        tag_vals=[sorted(tag_vals[i]) for i in range(len(tag_vals))]
        output_shape=ds.pixel_array.shape
        for i in range(len(tag_list)-1):
            output_shape=output_shape+(len(tag_vals[i]),)
        output_array=np.zeros(output_shape)
        for ds in output:
            for tag in tag_list[1:]:                #assign ds.pixel_array to correct indices
                output_array[(slice(None),slice(None))+tuple([tag_vals[tag_list[1:].index(tag)].index(ds.__dict__[tag]) for tag in tag_list[1:]])]=ds.pixel_array
        outputs.append(output_object(tag_vals,output_array,tag_names[1:],refslices_list[UID_list.index(UID)]))     
        if verbose==True:
            print('MESSAGE: Created a',output_array.shape,'array for series UID',UID,'out of a total of',len(output),'files for that series.\n')
            if np.prod(output_array.shape[2:]) != len(output):
                print('WARNING:',np.prod(output_array.shape[2:]),'slices were created out of',len(output),'files. Are you sure the specified dicom tags are discriminating enough?\n')
    return outputs