#!/bin/bash

# Ensure output directories exist
mkdir -p results logs

# Loop through EXACT file fractions
for frac in 2_5p 5p 10p 20p; do
    
    # Define file names
    INPUT_FILE="subsamples/DM_particles_ss${frac}_250cube.dat"
    OUTPUT_NAME="bitvf_aletheia_${frac}"
    JOB_FILE="submit_${frac}.job"

    # Generate the SLURM script
    cat <<EOF > ${JOB_FILE}
#!/bin/bash
#SBATCH --job-name=BitVF_${frac}
#SBATCH --account=astro
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --constraint=blade
#SBATCH --output=logs/bitvf_${frac}_%j.out

echo "Executing BitVF for fraction: ${frac}"
./run_aletheia_bitvf ${INPUT_FILE} ${OUTPUT_NAME}
EOF

    # Submit the job
    echo "Submitting ${JOB_FILE}..."
    sbatch ${JOB_FILE}

done