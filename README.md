## ChemiMap
# Problem & Motivation
"Reading through hundreds of scientific papers is so fun and efficient!" said no one, ever. 

A researcher in the biochemical field typically spends hundreds of hours sifting through vast amount of scientific papers in search of information about a chemical, a disease, and their relationship. Exploring complex chemical-disease relationships is tough, and although there exist tools and databases for researchers, it's very difficult to visualize these relationships efficiently as they have to find and read through these articles in order to find the information that they are looking for. Researchers have to sift through fragmented and time-consuming material, and they often explore multiple hypotheses simultaneously, making it challenging to efficiently cross-reference large volumes of material. In the time spent reading, important connections get lost, and the process is slow and overwhelming. 

# Our Solution
Introducing ChemiMap, a chemical-disease relationship explorer that allows the user to quickly find relationships and relevant papers across vast amounts of literature. We designed a product to make research more efficient. ChemiMap uses state-of-the-art Named Entity Recognition (NER) and Relation Extraction (RE) models to find chemicals and diseases in papers and analyzes the most salient relationships between them. This information is presented in an easy-to-visualize interactive knowledge graph along with annotated abstracts of papers highlighting the entities, making research quicker and more efficient. With ChemiMap, we estimate an annual saving of $4 million in total with a user base of 1,000 researchers. 

# Data Source and Data Science Approach
The BioCreative V Chemical Disease Relation (CDR) dataset is a text corpus of human annotations of chemicals, diseases, and interactions in 1,500 PubMed articles. This dataset is used to train our two core models: the ConNER NER model and the SSGU-CD RE model. 
