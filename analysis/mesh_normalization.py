# MeSH Normalization Script
# Run after NER output (NER Post-processing) to normalize the entities into correctly mapped MeSH IDs
# In case there isn'a a match, "unmatched_xxx" is written
# Output from this script will be input into RE (RE Pre-processing)

import pandas as pd
from collections import defaultdict

# Load the lookup table and create a reverse dictionary for entity matching
def load_lookup_table(file_path):
    """
    Load mesh_lookup_table_with_dsstox.csv and create a reverse dictionary for ID matching
    """
    mesh_lookup_df = pd.read_csv(file_path)
    # Expand entries for each term, lowercase for case-insensitivity
    mesh_lookup_expanded = mesh_lookup_df.assign(
        names=mesh_lookup_df['Names/Entry Terms'].str.split('|')
    ).explode('names')
    mesh_lookup_expanded['names'] = mesh_lookup_expanded['names'].str.strip().str.lower()
    
    # Create a reverse lookup dictionary
    term_to_mesh = pd.Series(mesh_lookup_expanded['MeSH ID'].values, index=mesh_lookup_expanded['names']).to_dict()
    return term_to_mesh

# Parse the NER output text file
def parse_ner_output(file_path):
    """
    Parse NER output txt file
    """
    papers = defaultdict(lambda: {"title": "", "abstract": "", "entities": []})
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Check if it's a title or abstract line
            if "|t|" in line or "|a|" in line:
                paper_id, section, text = line.split("|", 2)
                if section == "t":
                    papers[paper_id]["title"] = text
                elif section == "a":
                    papers[paper_id]["abstract"] = text
            else:
                # This is an entity line
                parts = line.split("\t")
                if len(parts) == 6:
                    paper_id, start, end, entity, entity_type, entity_num = parts
                    papers[paper_id]["entities"].append({
                        "start": int(start),
                        "end": int(end),
                        "name": entity,
                        "type": entity_type,
                        "entity_num": entity_num
                    })
    return papers

# Map entities to MeSH IDs or placeholders
def map_entities(papers, lookup_table):
    """
    Map entities from the parsed ner output to correct MeSH IDs
    """
    unmatched_placeholder = {}  # Store placeholders for unmatched terms
    placeholder_id = 1  # Starting ID for unmatched terms

    for paper in papers.values():
        for entity in paper["entities"]:
            name_lower = entity["name"].lower()
            if name_lower in lookup_table:
                entity["entity_num"] = lookup_table[name_lower]
            else:
                # Assign consistent placeholder for unmatched entities
                if name_lower not in unmatched_placeholder:
                    unmatched_placeholder[name_lower] = f"unmatched_{placeholder_id}"
                    placeholder_id += 1
                entity["entity_num"] = unmatched_placeholder[name_lower]
    
    return papers

# Write the modified output back to a text file in the original format
def write_output(papers, output_path):
    """
    Output into the txt file
    """
    with open(output_path, 'w') as f:
        for paper_id, paper in papers.items():
            # Write title and abstract lines
            if paper["title"]:
                f.write(f"{paper_id}|t|{paper['title']}\n")
            if paper["abstract"]:
                f.write(f"{paper_id}|a|{paper['abstract']}\n")
            
            # Write each entity line with updated entity_num
            for entity in paper["entities"]:
                f.write(f"{paper_id}\t{entity['start']}\t{entity['end']}\t{entity['name']}\t{entity['type']}\t{entity['entity_num']}\n")
            
            # Add an empty line after each paper
            f.write("\n")

# Main function
def main(input_file, lookup_file, output_file):
    """
    Run MeSH Normalization script in whole
    """
    # Load the lookup table to create the reverse lookup dictionary
    lookup_table = load_lookup_table(lookup_file)
    
    # Parse the NER output file
    papers = parse_ner_output(input_file)
    
    # Map entities to their official names or placeholders
    papers = map_entities(papers, lookup_table)
    
    # Write the output to a new file with updated entity numbers
    write_output(papers, output_file)

# Run
if __name__ == "__main__":
    # When running .py file, below can be commented out and instead be defined outside by .py call 
    mesh_normalization_input_file = "ner_output.txt"  # Replace with your input NER output file
    lookup_file = "mesh_lookup_table_with_dsstox.csv" 
    output_file = "mesh_normalized_ner_output.txt"  # Output file path
    
    main(mesh_normalization_input_file, lookup_file, output_file)
