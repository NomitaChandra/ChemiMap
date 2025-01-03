# Run Pipeline Script

This `run_pipeline.sh` script automates the process of running a multi-step data processing pipeline on an EC2 instance. It includes steps to activate environments, execute Python scripts, and manage output files.

## Prerequisites

Before running the script, ensure the following are set up:
1. **Python Environment**: 
   - A virtual environment (`venv`) with all required dependencies installed.
   - A Conda environment named `ssgu_env` with required dependencies.
2. **Pipeline Scripts**:
   - `pipeline_combined.py` 
   - `process_and_evaluate.py` 
3. **Input File**:
   - Ensure you have a valid input file in CSV format.

## Script Usage

### Command
Run the script with the following command:

```bash
./run_pipeline.sh <input_file_path> <output_file_name>
```
### Arguments
- <input_file_path>: Full path to the input CSV file.
- <output_file_name>: Desired name for the intermediate output file.

## Script Workflow
- Navigate to SSGU-CD-all Directory:

- Changes directory to ~/SSGU-CD-all.
- Activate Virtual Environment:

- Activates the virtual environment (venv).
- Run Part 1 of the Pipeline:

  - Changes directory to ~/capstone-chemimed/Justin_working_file.
  - Executes pipeline_combined.py with the provided input and output file arguments.
  - Moves the generated output file to ~/SSGU-CD-all/dataset/cdr.
  - Rename Output File:

  - Renames the output file to CDR_TestSet.PubTator.txt.
- - Run Part 2 of the Pipeline:

- Deactivates the virtual environment and activates the Conda environment ssgu_env.


  - Executes process_and_evaluate.py in the ~/SSGU-CD-all directory.

## Output
- Intermediate output: Saved in ~/SSGU-CD-all/dataset/cdr as CDR_TestSet.PubTator.txt.
0 Final output: Generated by process_and_evaluate.py.

## Error Handling
The script checks for the existence of directories and files before proceeding.

If any step fails, it outputs an error message and stops execution.
Notes

Ensure all dependencies are installed in the respective environments.

Adjust paths in the script if your directory structure differs.

## Troubleshooting
Parser Error: Check your input file for format issues (e.g., inconsistent columns).

Permission Error: Ensure the script has execute permissions:
``` 
chmod +x run_pipeline.sh
```

Environment Activation Issues: Verify that venv and ssgu_env are correctly set up and accessible.

# Pre-Processing
`pipeline_combined.py`

This script levarages a complex pre-processing structure based on the scripts below.

## pipeline_part_1.py: 
### Text Preprocessing and Entity Extraction for ConNER

This script prepares and processes text data for Named Entity Recognition (NER) using the ConNER model. It takes input data in CSV format, preprocesses it, and extracts entities like chemicals and diseases. The results are formatted for further use in relation extraction.

### Features

1. **Input Preprocessing**:
   - Reads a CSV file containing `title`, `abstract`, and `article_code`.
   - Merges `title` and `abstract` into a single text field.

2. **Conversion to ConNER Format**:
   - Tokenizes and converts text to a format compatible with the ConNER model.
   - Maps words to token indices for precise entity alignment.

3. **NER Model Inference**:
   - Loads a pre-trained ConNER model (e.g., based on RoBERTa).
   - Performs Named Entity Recognition to identify chemicals and diseases.

4. **Post-Processing**:
   - Regularizes and validates model predictions to ensure consistency.
   - Extracts entities and their positions in the text.

5. **Output Formatting**:
   - Outputs extracted entities in a format suitable for relation extraction (RE) models.

### Prerequisites

- Python 3.x with the following libraries installed:
  - `pandas`, `torch`, `transformers`, `numpy`, `scipy`, `bs4`, and `tqdm`.
- Pre-trained ConNER model stored in the `./ConNER` directory.
- Dependencies:
  - `data_utils.py`, `flashtool.py`, and `ConNER_model_definition.py` (from the ConNER repository).
- Input data in the `./data/` directory.


### Command
Run the script with the following command:
```bash
python pipeline_part_1.py --input_file <input_csv_path> --output_file <output_txt_path>
```

## pipeline_part_2_mesh.py:
### MeSH Normalization Script

This script performs MeSH normalization on Named Entity Recognition (NER) output, mapping entities like chemicals and diseases to their corresponding MeSH IDs. For entities without a match, placeholder IDs (`unmatched_xxx`) are assigned. The output from this script is used as input for Relation Extraction (RE) pre-processing.

### Features

1. **MeSH Lookup Table**:
   - Loads a lookup table (`mesh_lookup_table_with_dsstox.csv`) and creates a reverse dictionary to map entity names to MeSH IDs.
   - Handles case-insensitivity and supports multiple terms for a single MeSH ID.

2. **NER Output Parsing**:
   - Reads and parses the NER output file.
   - Extracts entities along with their metadata (start/end positions, type, etc.).

3. **Entity Mapping**:
   - Maps entities to their corresponding MeSH IDs.
   - Assigns placeholder IDs for unmatched entities.

4. **Output Formatting**:
   - Writes the normalized output back to a text file, maintaining the original format with updated MeSH IDs.

### Prerequisites

- Python 3.x with the following libraries installed:
  - `pandas`, `argparse`, `collections`.
- A valid lookup table file (`mesh_lookup_table_with_dsstox.csv`).
- Input NER output file in the required format.


#### Command
Run the script with the following command:
```bash
python pipeline_part_2_mesh.py --input_file <ner_output_file> --lookup_file <lookup_table_file> --output_file <normalized_output_file>
```

# Relation Extraction (RE) and Post processing
`process_and_evaluate.py`

This part leverages the scripts in ChemiMap/models/SSGU-CD-all.

Workflow

- Pre-processing: Prepare input files using process_cdr.py.
- Model Training and Evaluation: Run process_and_evaluate.py to train the model and generate predictions.
- Post-processing: Use re_postprocessing.py to structure RE outputs and map entities.



## process_and_evaluate.py
- Trains and evaluates the RE model using transformer-based architectures.
- Includes pre-processing, training, and evaluation functions.

Key Features:

- Configurable model parameters (e.g., transformer type, learning rate).
- GPU or CPU support.
- Outputs predictions in .pubtator format for compatibility with existing tools.


## re_postprocessing.py 
- Processes RE outputs into structured CSV files.
- Maps MeSH IDs to official names using a lookup table.
- Outputs:
  - `re_output_postprocessed_entities.csv`: Entities with normalized names.
  - `re_output_postprocessed_relationships.csv`: Relationships between entities.

#### Usage:
Replace paths for the input RE file, MeSH lookup file, and output CSVs, then run the script:

## evaluation.py
- Evaluates RE model predictions against gold-standard data.
- Metrics: Precision, Recall, and F1-Score.
- Converts predictions into formats like .pubtator for further analysis.


## License
This script is provided "as-is" without warranty of any kind.