import subprocess
import glob

#input = 'input_list'  # Replace with the path to the input folder
with open('input_list_g.txt') as file:
    for line in file:
# Set the input folder name
        input_folder = line.strip()

# Construct the input and output file names
        input_f_name = f"/projectnb/bucbi/GW_Georgetown/{input_folder}"
        input_file= glob.glob(f"{input_f_name}/MPRAGE*")[0]
        print(input_file,"Hi")
        output_file = f"{input_file.rstrip('.nii')}_brain"

# Construct the command
        command = ["/share/pkg.7/fsl/6.0.4/install/bin/bet", input_file , output_file, "-R", "-f", "0.6", "-g", "0"]

# Execute the command
        subprocess.run(command)