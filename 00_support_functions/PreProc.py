
import os, sys, glob
import re
import json
import subprocess
from pathlib import Path

from pyfunctions import *

def bmri_ai_prep_pipeline(datadir, flag="test"):
    # grab all *.niis in data directory
    subjects_path = os.path.join(datadir, "sub-*")
    subjects = glob.glob(subjects_path)
    sessions_path = os.path.join(subjects_path, "ses-*")
    sessions = glob.glob(sessions_path)
    series_path = os.path.join(sessions_path, "*.nii")
    series = glob.glob(series_path)

    # run pipeline according to run flag
    if flag == "test":
        for s in tqdm(series):
            PreProc(s).print()
    elif flag == "run":
        for s in tqdm(series):
            a_series = PreProc(s)
            a_series.runPipeline()
    else:
        print("unknown flag ['test'/'run'], exiting ...")
        return
    print("all done! thank you for using bmri-ai-prep!")
    return

class PreProc:
    def __init__(self, path):
        #print("processing: ", path)
        self.pipeline = []
       
        self.path = path
        split_path = path.split("/")
        subRe = re.compile('sub-*')
        self.subId = list(filter(subRe.match, split_path))[0]
        subIdIdx = split_path.index(self.subId)

        self.dataDir = ""
        for i in range(subIdIdx):
            self.dataDir = os.path.join(self.dataDir, split_path[i])
        print("datadir: ", self.dataDir)
        sesRe = re.compile('ses-*')
        self.sesId = list(filter(sesRe.match, split_path))[0]


        self.filename = split_path[-1]
        self.basename = os.path.splitext(self.filename)[0]
        split_filename = self.basename.split("_")
        self.typId = split_filename[2]      
        if "cpp" in self.filename.lower():
            print("this is a deformation registration transform file, skipping ...")
            self.typId = "transform"
            self.isRef = False
            self.refPath = ""
            self.nt = 0
            self.pipeline = []
            return

        # check if the series is a reference file, and if not, try to grab the reference file 
        if 'ref' in self.typId:
            self.isRef = True
        else:
            self.isRef = False
            self.refPath = path.replace(self.filename, self.subId+"_"+self.sesId+"_ref.nii")
            if not os.path.exists(self.refPath):
                print("ERROR! reference file cannot be found!")

        # using afni program to grab the NT of the series
        self.nt = int(subprocess.run(['3dinfo', '-nt', self.path], stdout=subprocess.PIPE).stdout.decode('utf-8').strip())

        # trying to find bmri-prep processing pipeline (baip) file
        self.baipPath = self.path.replace('.nii', '.baip.txt')
        if os.path.exists(self.baipPath):
            print("baip.json file found, parsing it")
            with open(self.baipPath) as baip_file:
                baip_data = json.load(baip_file)
                self.pipeline = baip_data
        else:
            print("making new baip.json")
            if self.nt > 1 and 'moco' not in self.basename.lower():
                print("this is a 3D+t dataset, and not already motion corrected, so it will be done.")
                self.pipeline.insert(0, "MOCO")
            if not self.isRef:
                self.pipeline.append("REG")
            else:
                self.pipeline.append("SEG")
            with open(self.baipPath, 'w') as output_file:
                print(json.dumps(self.pipeline, indent=2), file=output_file)

    def runPipeline(self):
        if not self.pipeline:
            print("nothing to be done, making direct copy to derived folder")

            
        self.cur_proc_path = self.path
        for op in tqdm( range(len(self.pipeline)) ):
            an_op = self.pipeline.pop(0)
            
            if an_op == "MOCO":
                split_cur_path = self.cur_proc_path.split("/")
                current_working_dir = "/".join(split_cur_path[:-1])
                Path(current_working_dir).mkdir(parents=True, exist_ok=True)
                
                output_filename = os.path.splitext(split_cur_path[-1])[0]+'_Moco'
                output_path = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename)

                print("Perform motion correction here")
                print("Input filename: ", self.cur_proc_path)
                print("Output filename: ", output_path)
                moco_nrdef(self.cur_proc_path, output_path, ref='average')

                self.cur_proc_path = output_path+".nii"
                with open(self.baipPath, 'w') as output_file:
                    print(json.dumps(self.pipeline, indent=2), file=output_file)
            elif an_op == "REG":
                split_cur_path = self.cur_proc_path.split("/")
                current_working_dir = "/".join(split_cur_path[:-1])
                Path(current_working_dir).mkdir(parents=True, exist_ok=True)

                output_filename = os.path.splitext(split_cur_path[-1])[0]+'_Reg.nii'
                transform_filename = os.path.splitext(split_cur_path[-1])[0]+'_Reg.cpp.nii'
                output_path = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename)
                transform_path = os.path.join(self.dataDir, "derived", self.subId, self.sesId, transform_filename)
                
                print("Perform coregistration here")
                print("Reference filename: ", self.refPath)
                print("Input filename: ", self.cur_proc_path)
                print("Output filename: ", output_path)
                print("Transform filename: ", transform_path)
                coreg_nrdef(self.refPath, self.cur_proc_path, output_path, transform_path)
                
                self.cur_proc_path = output_path
                with open(self.baipPath, 'w') as output_file:
                    print(json.dumps(self.pipeline, indent=2), file=output_file)
            elif an_op == "SEG":
                split_cur_path = self.cur_proc_path.split("/")
                current_working_dir = "/".join(split_cur_path[:-1])
                Path(current_working_dir).mkdir(parents=True, exist_ok=True)

                output_filename_mask = os.path.splitext(split_cur_path[-1])[0]+'_Mask.nii'
                output_filename_segl = os.path.splitext(split_cur_path[-1])[0]+'_Segl.nii'
                output_filename_maskl = os.path.splitext(split_cur_path[-1])[0]+'_Maskl.nii'
                output_filename_segr = os.path.splitext(split_cur_path[-1])[0]+'_Segr.nii'
                output_filename_maskr = os.path.splitext(split_cur_path[-1])[0]+'_Maskr.nii'
                output_path_mask = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename_mask)
                output_path_segl = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename_segl)
                output_path_maskl = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename_maskl)
                output_path_segr = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename_segr)
                output_path_maskr = os.path.join(self.dataDir, "derived", self.subId, self.sesId, output_filename_maskr)
                
                print("Perform anatomy segmentation here")
                print("Input filename: ", self.cur_proc_path)
                print("Output filename: ", output_path_mask)
                seg_vit(self.cur_proc_path, output_path_mask, output_path_maskl, output_path_maskr, output_path_segl, output_path_segr)

                self.cur_proc_path = output_path_mask
                with open(self.baipPath, 'w') as output_file:
                    print(json.dumps(self.pipeline, indent=2), file=output_file)
            else:
                print("Unknown pipeline operation! Panic exiting!")
                return

    def print(self):
        print("path to file: ", self.path)
        print("subject-ID: ", self.subId)
        print("session-ID: ", self.sesId)
        print("series type: ", self.typId)
        if self.isRef:
            print("IS REFERENCE")
        else:
            print("reference path: ", self.refPath)
        print("pipeline: ", self.pipeline)
        print("")