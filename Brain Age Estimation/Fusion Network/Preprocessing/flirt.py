import subprocess
import glob

#input = 'input_list'  # Replace with the path to the input folder
with open('input_list_g_new.txt') as file:
    for line in file:
# Set the input folder name
        input_folder = line.strip()

# Construct the input and output file names
        input_f_name = f"/projectnb/bucbi/GW_Georgetown/{input_folder}"
        input_file= glob.glob(f"{input_f_name}/*_brain.nii.gz")[0]
        print(input_file,"Hi")
        output_file = f"{input_file.rstrip('.nii.gz')}_MNI1mmbrainfinal.nii.gz"
        ref_file = '/share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_1mm_brain'
        mat_file=f"{output_file.rstrip('.nii.gz')}.mat"

# Construct the command
#/share/pkg.7/fsl/6.0.4/install/bin/flirt -in /projectnb/bucbi/Gulf_war_T1_T2/1002/MPRAGE_SENSE2_301_MPRAGE_SENSE2_SENSE_20150430131322_301_brain.nii.gz -ref /share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_2mm_brain -out /projectnb/bucbi/Gulf_war_T1_T2/1002/brain_MNI1mm.nii.gz -omat /projectnb/bucbi/Gulf_war_T1_T2/1002/brain_MNI1mm.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear
        command = ['/share/pkg.7/fsl/6.0.4/install/bin/flirt', 
           '-in', input_file,
           '-ref', ref_file,
           '-out', output_file,
           '-omat', mat_file,
           '-bins', '256',
           '-cost', 'corratio',
           '-searchrx', '-90', '90',
           '-searchry', '-90', '90',
           '-searchrz', '-90', '90',
           '-dof', '12',
           '-interp', 'trilinear']

# Execute the command
        subprocess.run(command)