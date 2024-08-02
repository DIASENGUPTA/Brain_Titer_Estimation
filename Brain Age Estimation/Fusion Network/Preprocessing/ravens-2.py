import os
import glob
import nibabel as nib
import numpy as np

#input = 'input_list'  # Replace with the path to the input folder
with open('input_list_g.txt') as file:
    for line in file:
# Set the input folder name
        input_folder = line.strip()

# Construct the input and output file names
        path=f"/projectnb/bucbi/GW_Georgetown"
        subj=input_folder
        input_f_name = f"/projectnb/bucbi/GW_Georgetown/{input_folder}"
        input_file= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz")[0]
        print(input_file,"Hi")
        binary_mask = f"{path}/{subj}/binary_mask.nii.gz"
        #output_file = f"{input_file.rstrip('.nii.gz')}_MNI1mm_to_subjFS.nii.gz"
        #ref_file = '/share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_2mm_brain'
        #mat_file=f"{output_file.rstrip('.nii.gz')}.mat"
        # Convert binary mask to unsigned char data type
        img = nib.load(binary_mask)
        data = img.get_fdata()
        data = np.uint8(data)
        img = nib.Nifti1Image(data, img.affine, img.header)
        img.set_data_dtype(np.uint8)
        nib.save(img, binary_mask)

        text = f"""
#!/bin/bash -l

# Request 16 CPUs
#$ -pe omp 16

# Request 1 GPU 
#$ -l gpus=1

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=6.0

# Email when done
#$ -m ea

# Combine output and error files into a single file
#$ -j y

# 1 hour
#$ -l h_rt=1:00:00

/projectnb/bucbi/dramms-1.5.1/bin/dramms -S {input_file} -T /share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_1mm_brain.nii.gz\
 -O {path}/{subj}/MNI1mm_to_subjFS.nii.gz -D {path}/{subj}/def_MNI1mm_to_subjFS.nii.gz -R {path}/{subj}/ravens_in_MNI1mm\
 -L {binary_mask} -l 1
        """

        #command = ['/projectnb/bucbi/dramms-1.5.1/bin/dramms', 
        #   '-S', input_file,
        #   '-T', '/share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_1mm_brain.nii.gz',
        #   '-O', f"{path}/{subj}/MNI1mm_to_subjFS.nii.gz",
        #   '-D', f"{path}/{subj}/def_MNI1mm_to_subjFS.nii.gz",
        #   '-R', f"{path}/{subj}/ravens_in_MNI1mm",
        #   '-L', binary_mask,
        #   '-l', '1']

        #subprocess.run(command)
        with open("script.sh", "w") as f:
                f.write(text)
        os.system(f"qsub -N J_{input_folder} -P bucbi script.sh")