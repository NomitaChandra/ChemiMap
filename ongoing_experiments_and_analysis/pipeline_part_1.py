# pipeline_part_1.py
# ConNER: Text Preprocessing & Output Processing

"""
This script prepares a pipeline that:
(i) preprocesses incoming text (abstracts) and updates them to the format expected by ConNER;
(ii) processes the output of the model to extract the entities.

Dependencies:
- `data_utils.py` (from the ConNER repository)
- `flashtool.py` 
- `ConNER_model_definition.py` (from the ConNER repository)
- A pre-trained ConNER model stored in the `./ConNER` directory
- Input data files in the `./data/` directory
"""

# Section 1: Imports

import os
import bs4
import numpy as np
import scipy
import json
import pandas as pd
import re
import string
from pprint import pprint

# Model and Tokenization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, KLDivLoss

from transformers import (
    BertPreTrainedModel,
    BertForTokenClassification,
    BertModel,
    RobertaModel,
    RobertaTokenizer,
    RobertaConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertConfig,
)

# Evaluation Related
import argparse
import logging
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Custom Modules (Ensure these are available in your PYTHONPATH or the same directory)
from data_utils import (
    tag_to_id,
    get_chunks,
    get_labels,
    convert_examples_to_features,
    InputExample,
)
from flashtool import Logger
from ConNER_model_definition import RobertaForTokenClassification_v2

logger = logging.getLogger(__name__)

def main(input_file_path, output_file_path):
    # Section 2: Loading the Data

    """
    Expects incoming text with the fields "title" and "abstract".

    Loads the processed CSV files.

    If we want to use data from the PubMed database directly, we will have to build a processing pipeline for that.
    """

    # Paths to the datasets
    # Using only the test_path as the input_file_path
    test_path = input_file_path

    # Reading the file and retaining only the title, abstract, and article_code columns
    df_test = pd.read_csv(test_path)[['title', 'abstract', 'article_code']]

    # Forming a new column with the merged texts
    df_test['text'] = df_test["title"] + " " + df_test["abstract"]

    # This will be the starting point for further preprocessing.

    # Section 3: Converting Text to ConNER Format

    def convert_text_to_ConNER_format(df, tokenizer):
        '''
        Takes in a dataframe and returns an "example" object that can be taken by the
        "load_and_cache_examples" function from the data_utils.py module from the ConNER repo.
        Also outputs a mapping dictionary linking the word indices to token indices.

        Inputs:
        - df: DataFrame with a "text" column containing the combined title and abstract of a journal,
              as well as the PubMed ID ('article_code').
        - tokenizer: The tokenizer used in the NER model.

        Output:
        - examples: List of InputExample objects.
        - mapping_dict: Dictionary linking words to token indices.

        Other Prerequisites:
        - InputExample function, imported from the data_utils.py module.
        '''
        mode = "doc_dev"  # Stands for document-based evaluation for the dev set.

        texts = df['text']
        guid_index = 1
        examples = []
        mapping_dict = []

        for text in texts:

            example_dict = {}  # Mapping info for the current example (i.e., "text") only.

            # Handling the words. For handling punctuation, referenced:
            # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
            words = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation})).split()
            labels = [0] * len(words)  # Set labels to 0 as a dummy since we are doing inference.

            token_index = 1   # Starts from 1 because token 0 is the CLS token for BERT and RoBERTa.
            word_index = 0    # Counts from 0, as words in the SciBERT model are also counted from 0.

            # Prepares the word index
            for word in words:
                # Tokenization to see the length
                tokenized_word = tokenizer(word)['input_ids'][1:-1]  # Skips the 1st & last input ids (CLS and SEP tokens)
                num_of_tokens = len(tokenized_word)  # Number of sub-word tokens that the current word has

                assert len(example_dict) == word_index, "Error in code, word index is probably wrong"

                # Append a sub-list showing the list of corresponding token ids
                token_indices = list(range(token_index, token_index + num_of_tokens))
                example_dict[word_index] = {'word': word, 'token_idx': token_indices}

                word_index += 1
                token_index += num_of_tokens

            mapping_dict.append(example_dict)

            hp_labels = [None] * len(labels)  # Since "tags_hp" not in labels, set to None

            examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                         words=words,
                                         labels=labels,
                                         hp_labels=hp_labels))
            guid_index += 1

        return examples, mapping_dict

    # Load tokenizer
    model_path = "./ConNER/bc5cdr"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Convert text to ConNER format
    test_set, word_to_token_map = convert_text_to_ConNER_format(df_test, tokenizer)

    # Section 4: Dataset Formatting and Model Loading

    def load_and_cache_examples(args, df, tokenizer, labels, pad_token_label_id, mode,
                                entity_name='bc5cdr', remove_labels=False):
        examples, word_to_token_map = convert_text_to_ConNER_format(df, tokenizer)
        features = convert_examples_to_features(
            examples,
            labels,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # XLNet has a CLS token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # RoBERTa uses an extra separator between pairs of sentences
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # Pad on the left for XLNet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
            entity_name=entity_name,
        )

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        all_full_label_ids = torch.tensor([f.full_label_ids for f in features], dtype=torch.long)
        all_hp_label_ids = torch.tensor([f.hp_label_ids for f in features], dtype=torch.long)
        all_entity_ids = torch.tensor([f.entity_ids for f in features], dtype=torch.long)
        if remove_labels:
            all_full_label_ids.fill_(pad_token_label_id)
            all_hp_label_ids.fill_(pad_token_label_id)
        all_ids = torch.tensor([f for f in range(len(features))], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                                all_full_label_ids, all_hp_label_ids, all_entity_ids, all_ids)

        return dataset, word_to_token_map

    # Initialize device and parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_token_label_id = CrossEntropyLoss().ignore_index
    labels = ['O', 'B-Chemical', 'B-Disease', 'I-Chemical', 'I-Disease']

    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    args.model_type = "roberta"
    args.model_name_or_path = "./ConNER/bc5cdr"
    args.max_seq_length = 512   # Modified from 128
    args.per_gpu_train_batch_size = 8
    args.per_gpu_eval_batch_size = 8
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    args.local_rank = -1

    args.gradient_accumulation_steps = 1
    args.learning_rate = 5e-5
    args.weight_decay = 0.0
    args.adam_epsilon = 1e-8
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.98
    args.max_grad_norm = 1.0
    args.num_train_epochs = 3.0
    args.max_steps = -1
    args.warmup_steps = 0
    args.logging_steps = 10000
    args.save_steps = 10000
    args.seed = 1

    # Load and cache examples
    eval_dataset, word_to_token_map = load_and_cache_examples(
        args, df_test, tokenizer, labels, pad_token_label_id, mode="doc_dev"
    )

    # Create DataLoader
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Section 5: Model Inference

    # Load model
    test_model = RobertaForTokenClassification_v2.from_pretrained(model_path)
    test_model.to(device)

    # Set model to evaluation mode
    test_model.eval()

    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don't use segment_ids
            outputs = test_model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)

    # Section 6: Post-processing Predictions

    def regularize_results(preds, word_to_token_map, voting=False):
        '''
        Takes in the model predictions and regularizes them.

        Specifically, the classification of each token is checked to ensure that:
        (i)  It is either all '0' (i.e., Outside);
        (ii) Begins with '2' and ends with all '4's;
        (iii) Begins with '1' and ends with all '3's.

        By default, all irregular results are switched to '0's. The alternative is to vote, which can be toggled on.

        Inputs:
        - preds: Model predictions.
        - word_to_token_map: List of dictionaries showing the words and their mapping to tokenized IDs.
        - voting: Whether to use voting to irregularize results. Default is False.

        Output:
        - preds_cor: Corrected predictions as a numpy array with dtype=int64.
        '''
        assert preds.shape[0] == len(word_to_token_map), "Number of samples in the predictions and mapping are different"

        preds_cor = preds.copy()

        for i in range(preds.shape[0]):
            last_word_class = "outside"
            pred = preds[i]
            mapping = word_to_token_map[i]

            for word_idx in mapping.keys():
                tokens = mapping[word_idx]['token_idx']
                word_pred = pred[tokens[0]:tokens[-1]+1]
                word_length = len(word_pred)

                if word_length == 0:
                    continue

                else:
                    # Possible correct sequences:
                    b_chem_seq = np.empty([word_length, ], dtype=np.int64)
                    i_chem_seq = np.empty([word_length, ], dtype=np.int64)
                    b_disease_seq = np.empty([word_length, ], dtype=np.int64)
                    i_disease_seq = np.empty([word_length, ], dtype=np.int64)
                    outside_seq = np.zeros([word_length, ], dtype=np.int64)

                    b_chem_seq.fill(3)
                    b_chem_seq[0] = 1

                    b_disease_seq.fill(4)
                    b_disease_seq[0] = 2

                    i_chem_seq.fill(3)
                    i_disease_seq.fill(4)

                    if np.array_equal(word_pred, b_chem_seq):      # Correct B-Chemical sequence
                        last_word_class = "b-chemical"
                        continue

                    elif np.array_equal(word_pred, b_disease_seq):  # Correct B-Disease sequence
                        last_word_class = "b-disease"
                        continue

                    elif np.array_equal(word_pred, i_chem_seq):     # I-Chemical sequence
                        if last_word_class not in ["b-chemical", "i-chemical"]:  # Should follow another B or I chemical
                            preds_cor[i][tokens[0]] = 1            # Change to a B-Chemical sequence
                            last_word_class = "b-chemical"
                            continue
                        else:
                            last_word_class = "i-chemical"
                            continue

                    elif np.array_equal(word_pred, i_disease_seq):  # I-Disease sequence
                        if last_word_class not in ["b-disease", "i-disease"]:    # Should follow another B or I disease
                            preds_cor[i][tokens[0]] = 2            # Change to a B-Disease sequence
                            last_word_class = "b-disease"
                            continue
                        else:
                            last_word_class = "i-disease"
                            continue

                    elif np.array_equal(word_pred, outside_seq):     # Outside sequence
                        last_word_class = "outside"
                        continue

                    elif not voting:                                 # Irregular sequence without voting
                        preds_cor[i][tokens[0]:tokens[-1]+1] = 0    # Set to 'O'
                        last_word_class = "outside"

                    else:                                            # Irregular sequence with voting
                        mode_result = scipy.stats.mode(word_pred, axis=None, keepdims=False)
                        mode_value = mode_result.mode

                        if mode_value == 0:
                            preds_cor[i][tokens[0]:tokens[-1]+1] = 0
                            last_word_class = "outside"
                            continue
                        elif mode_value == 1:
                            preds_cor[i][tokens[0]] = 1
                            preds_cor[i][tokens[1]:tokens[-1]+1] = 3
                            last_word_class = "b-chemical"
                            continue
                        elif mode_value == 2:
                            preds_cor[i][tokens[0]] = 2
                            preds_cor[i][tokens[1]:tokens[-1]+1] = 4
                            last_word_class = "b-disease"
                            continue
                        elif mode_value == 3:
                            if last_word_class not in ["b-chemical", "i-chemical"]:
                                preds_cor[i][tokens[0]] = 1
                                preds_cor[i][tokens[1]:tokens[-1]+1] = 3
                                last_word_class = "b-chemical"
                                continue
                            else:
                                preds_cor[i][tokens[0]:tokens[-1]+1] = 3
                                last_word_class = "i-chemical"
                        elif mode_value == 4:
                            if last_word_class not in ["b-disease", "i-disease"]:
                                preds_cor[i][tokens[0]] = 2
                                preds_cor[i][tokens[1]:tokens[-1]+1] = 4
                                last_word_class = "b-disease"
                                continue
                            else:
                                preds_cor[i][tokens[0]:tokens[-1]+1] = 4
                                last_word_class = "i-disease"
                                continue
        return preds_cor


    def extract_entity_and_word_location_3(preds, word_to_token_map, df):
        '''
        Takes in the model predictions from ConNER and the word_to_token_map to extract
        the identified chemicals and diseases.

        Inputs:
        - preds: Model predictions.
        - word_to_token_map: List of dictionaries showing the words and their mapping to tokenized IDs.
        - df: DataFrame that contains at least 'title' and 'abstract' columns of the incoming samples.

        Output:
        - samples_postprocessing: List of dictionaries containing extracted entities and their details.
        '''
        preds = regularize_results(preds, word_to_token_map, voting=True)

        assert preds.shape[0] == len(word_to_token_map), "Number of samples in the predictions and mapping are different"
        assert preds.shape[0] == len(df), "Number of samples in the predictions and the dataframe are different"

        samples_postprocessing = []

        for i in range(preds.shape[0]):
            sample_postprocessing = {}
            pred = preds[i]
            mapping = word_to_token_map[i]

            title = df['title'].iloc[i]
            abstract = df['abstract'].iloc[i]
            text = title + ' ' + abstract
            sample_postprocessing['title'] = title
            sample_postprocessing['abstract'] = abstract
            sample_postprocessing['entities'] = []

            current_entity_type = None
            current_entity = None
            str_index = 0
            num_spaces = 0
            entity_index = 1
            previous_word = None
            word = None

            for word_idx in mapping.keys():
                word = mapping[word_idx]['word']
                tokens = mapping[word_idx]['token_idx']

                word_pred = pred[tokens[0]:tokens[-1]+1]

                # Loop through the full text to count the number of spaces
                while not text.startswith(word):
                    if text.startswith(" "):
                        num_spaces += 1
                        text = text[1:]
                    else:
                        error_message = f'The remaining text of sample {i} starts with "{text[:5]}...", ' \
                                        f'which does not start with the word "{word}" of word index {word_idx} or a space.\n' \
                                        f'The results so far: {sample_postprocessing["entities"]}'
                        raise ValueError(error_message)

                # Cases when a starting word is found:
                if 2 in word_pred and 1 not in word_pred:
                    assert 3 not in word_pred, f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as B-Disease but contains I-Chemical"

                    # Handles the case when a new entity immediately follows another entity
                    if current_entity_type is not None:
                        entity_string_index = [entity_start, entity_end]
                        entity_name = "entity_" + str(entity_index)

                        sample_postprocessing['entities'].append([current_entity, current_entity_type,
                                                                  entity_string_index, entity_name])

                        entity_index += 1

                    current_entity_type = 'disease'
                    current_entity = word

                    entity_start = str_index + num_spaces
                    str_index += (len(word) + num_spaces)
                    entity_end = str_index

                    num_spaces = 0
                    text = text[len(word):]

                elif 1 in word_pred and 2 not in word_pred:
                    assert 4 not in word_pred, f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as B-Chemical but contains I-Disease"

                    # Handles the case when a new entity immediately follows another entity
                    if current_entity_type is not None:
                        entity_string_index = [entity_start, entity_end]
                        entity_name = "entity_" + str(entity_index)

                        sample_postprocessing['entities'].append([current_entity, current_entity_type,
                                                                  entity_string_index, entity_name])

                        entity_index += 1

                    current_entity_type = 'chemical'
                    current_entity = word

                    entity_start = str_index + num_spaces
                    str_index += (len(word) + num_spaces)
                    entity_end = str_index

                    num_spaces = 0
                    text = text[len(word):]

                # Cases when a middle word is found:
                elif 2 not in word_pred and 4 in word_pred:
                    assert 3 not in word_pred, f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as having multiple classes"
                    assert current_entity_type == 'disease', f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as disease but follows a previous word of {current_entity_type} class"

                    current_entity = current_entity + ' ' * num_spaces + word
                    str_index += (len(word) + num_spaces)
                    entity_end = str_index

                    num_spaces = 0
                    text = text[len(word):]

                elif 1 not in word_pred and 3 in word_pred:
                    assert 4 not in word_pred, f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as having multiple classes"
                    assert current_entity_type == 'chemical', f"Error in prediction for sample {i}, the word '{word}' with predictions {word_pred} predicted as chemical but follows a previous word of {current_entity_type} class"

                    current_entity = current_entity + ' ' * num_spaces + word
                    str_index += (len(word) + num_spaces)
                    entity_end = str_index

                    num_spaces = 0
                    text = text[len(word):]

                # Cases when an outside word is found:
                elif len(word_pred) > 0 and word_pred[0] == 0:
                    if current_entity_type is not None:
                        entity_string_index = [entity_start, entity_end]
                        entity_name = "entity_" + str(entity_index)

                        sample_postprocessing['entities'].append([current_entity, current_entity_type,
                                                                  entity_string_index, entity_name])

                        entity_index += 1

                    current_entity_type = None
                    current_entity = None

                    str_index += (len(word) + num_spaces)
                    num_spaces = 0
                    text = text[len(word):]

                else:
                    # Unexpected prediction case
                    text = text[len(word):]
                    continue

            samples_postprocessing.append(sample_postprocessing)

        return samples_postprocessing

    # Extract entities
    samples_postprocessing = extract_entity_and_word_location_3(preds, word_to_token_map, df_test)

    # Verify that the string indices match
    for result in samples_postprocessing:
        text = result['title'] + ' ' + result['abstract']
        for entity_results in result['entities']:
            assert entity_results[0] == text[entity_results[2][0]:entity_results[2][1]]
    print("All clear!")

    # Section 7: Output Formatting for Relation Extraction (RE)

    def text_for_RE(extracted_list, df, file_path):
        '''
        Outputs the extracted list of entities into the file expected by the pre-processing function
        of the RE model.

        Inputs:
        - extracted_list: List object from the "extract_entity_and_word_location_3" function.
        - df: DataFrame that contains at least 'title', 'abstract', and 'article_code' columns of the incoming samples.
        - file_path: File path for saving the completed file.

        Output:
        - None. The text file is saved to the specified file path.
        '''
        assert df.shape[0] == len(extracted_list), "Number of samples in extracted list and dataframe do not match."

        with open(file_path, "w") as f:
            for i in range(df.shape[0]):
                PMID = str(df.article_code.iloc[i])
                title = df.title.iloc[i]
                abstract = df.abstract.iloc[i]

                f.write(PMID + "|t|" + title + "\n")
                f.write(PMID + "|a|" + abstract + "\n")

                for entity in extracted_list[i]['entities']:
                    entity_name = entity[0]
                    entity_type = entity[1].capitalize()
                    start = str(entity[2][0])
                    end = str(entity[2][1])
                    MeSH = str(entity[3])

                    f.write(PMID + "\t" + start + "\t" + end + "\t"
                            + entity_name + "\t" + entity_type + "\t"
                            + MeSH + "\n")

                f.write("\n")

        print(f"Complete. Results for {df.shape[0]} samples outputted in the text file at '{file_path}'.")

        return None

    # Output the extracted entities to a text file
    text_for_RE(samples_postprocessing, df_test, file_path=output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Path to input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
