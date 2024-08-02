import subprocess
import glob

#input = 'input_list'  # Replace with the path to the input folder
with open('input_list_g.txt') as file:
    for line in file:
# Set the input folder name
        input_folder = line.strip()

# Construct the input and output file names
        path=f"/projectnb/bucbi/GW_Georgetown/"
        subj=input_folder
        input_f_name = glob.glob(f"/projectnb/bucbi/GW_Georgetown/{input_folder}")[0]
        input_file= glob.glob(f"{input_f_name}/*_MNI1mmbrainfinal.nii.gz*")[0]
        print(input_file,"Hi")
        output_file = f"{input_f_name}/binary_mask.nii.gz"

        command = ['fslmaths', input_file, '-thr', '0.5', '-bin', output_file]

        subprocess.run(command)