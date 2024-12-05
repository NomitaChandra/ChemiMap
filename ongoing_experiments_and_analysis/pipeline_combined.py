# pipeline_combined.py
import argparse
import pipeline_part_1
import pipeline_part_2_mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--lookup_file", type=str, default="mesh_lookup_table_with_dsstox.csv", help="Path to lookup CSV file")
    parser.add_argument("--output_file", type=str, help="Path to final output file")
    args = parser.parse_args()

    # Intermediate output file from pipeline_part_1
    intermediate_output_file = "output_1_NER.txt"

    # Run pipeline_part_1
    pipeline_part_1.main(args.input_file, intermediate_output_file)

    # Run pipeline_part_2_mesh
    pipeline_part_2_mesh.main(intermediate_output_file, args.lookup_file, args.output_file)

if __name__ == "__main__":
    main()
