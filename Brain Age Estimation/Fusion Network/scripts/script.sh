
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

/projectnb/bucbi/dramms-1.5.1/bin/dramms -S /projectnb/bucbi/GW_Georgetown/0579-058/MPRAGE_Day1_bra_MNI1mmbrainfinal.nii.gz -T /share/pkg.7/fsl/6.0.4/install/data/standard/MNI152_T1_1mm_brain.nii.gz -O /projectnb/bucbi/GW_Georgetown/0579-058/MNI1mm_to_subjFS.nii.gz -D /projectnb/bucbi/GW_Georgetown/0579-058/def_MNI1mm_to_subjFS.nii.gz -R /projectnb/bucbi/GW_Georgetown/0579-058/ravens_in_MNI1mm -L /projectnb/bucbi/GW_Georgetown/0579-058/binary_mask.nii.gz -l 1
        