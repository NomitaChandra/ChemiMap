#!/bin/bash

# Step 1: Navigate to the first directory
cd ~/SSGU-CD-all || { echo "Directory ~/SSGU-CD-all not found!"; exit 1; }

# Step 2: Activate the first environment
source venv/bin/activate || { echo "Failed to activate virtual environment!"; exit 1; }

# Step 3: Run part 1 of the pipeline
cd ~/capstone-chemimed/Justin_working_file || { echo "Directory ~/capstone-chemimed/Justin_working_file not found!"; exit 1; }

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2

python pipeline_combined.py --input_file "$INPUT_FILE" --output_file "$OUTPUT_FILE" || { echo "Failed to run pipeline_combined.py!"; exit 1; }

# Move the output file to the desired directory
mv "$OUTPUT_FILE" ~/SSGU-CD-all/dataset/cdr/ || { echo "Failed to move output file!"; exit 1; }

# Step 4: Rename the file
cd ~/SSGU-CD-all/dataset/cdr || { echo "Directory ~/SSGU-CD-all/dataset/cdr not found!"; exit 1; }
mv "$OUTPUT_FILE" CDR_TestSet.PubTator.txt || { echo "Failed to rename the file!"; exit 1; }

# Step 5: Exit the virtual environment and activate the Conda environment
deactivate || { echo "Failed to deactivate virtual environment!"; exit 1; }
source ~/miniconda3/etc/profile.d/conda.sh || { echo "Failed to source conda.sh!"; exit 1; }
conda activate ssgu_env || { echo "Failed to activate Conda environment ssgu_env!"; exit 1; }

# Step 6: Run part two of the pipeline
cd ~/SSGU-CD-all || { echo "Directory ~/SSGU-CD-all not found!"; exit 1; }
python3 process_and_evaluate.py || { echo "Failed to run process_and_evaluate.py!"; exit 1; }

echo "Pipeline executed successfully!"
