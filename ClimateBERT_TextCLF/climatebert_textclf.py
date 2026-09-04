# TO UPLOAD ## FINAL VERSION

# see v2 sentiment analysis for diff version that doesn't use chunking


## TEXT CLASSIFICATON using ClimateBERT 
# VER 2 using pipeline 

# input file = english_only_df.parquet stored in
## /.../Sampled_Files/Text_Extraction_Final/english_only_df.parquet

## using my gpu for this step

## want to classify each paragraph (text) (each row) into one of the
# climateBERT categories (labels) using the pretrained ClimateBERT model
## --> simply asking whether paragraph (text) is climate-related or not (binary classification)

## will drop paragraphs/texts/rows that are NOT climate-related
# --> only keeping climate-related paragraphs/texts/rows for the next step of the pipeline

# inferencing using climateBERT model (distilroberta-base-climate-detector) for text classification


## next step will be Sentiment Analysis using CliamteBERT again

# relevant documentation:
# https://huggingface.co/climatebert/distilroberta-base-climate-detector
# https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
# (full credits go to authors of ClimateBERT; cited in my paper)
# https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html

## model trained on dataset that has label 0 or 1 


import pandas as pd
import torch
import csv
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import time
from datasets import Dataset
import gc


OUTPUT_FILE = "/.../MRP/ClimateBERT_TextCLF/classified_paragraphs.parquet"

MODEL_NAME = "climatebert/distilroberta-base-climate-detector"
BATCH_SIZE = 128

INPUT_FILE = "/.../MRP/Sampled_Files/Extracted_Paragraphs_Final/english_only_df.parquet"

BATCH_CHECKPOINT = 100000 ## checking after paragraphs/texts/rows processed

CHECKPOINT_FILE = "/.../MRP/ClimateBERT_TextCLF/checkpoint_file_classified.csv"

## update to prevent crashing continuously due to OOM
CHUNK_SIZE = 50000

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", d)


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    # model.to(d)
    model.eval()

    print(model.config.id2label)

    return tokenizer, model




def process_batch(df, pipe, starting_row=0):


    # clf_label = []
    # score = []

    dataset = Dataset.from_pandas(df)

    start_time = time.time()

    print(f"\n\nSTARTING CLASSIFICATION for {len(df)} paragraphs/texts/rows\n\n")

    with open(CHECKPOINT_FILE, "a", newline="", buffering=1) as f:
        w = csv.writer(f)

        for i, out in enumerate(tqdm(pipe(KeyDataset(dataset, "text"), padding=True, truncation=True), total=len(dataset))):
            w.writerow([i + starting_row, out["label"], out["score"]])

            if (i + 1) % BATCH_CHECKPOINT == 0:
                print(f"\nprocessed {i + 1 + starting_row} paragraphs/texts/rows; total run time: {(time.time() - start_time)/60:.2f} minutes\n")

    
    print("\nDONE CLASSIFYNG CHUNK (CLIMATEBERT DETECTION) ")
    print(f"\ntotal run time: {(time.time() - start_time)/60:.2f} minutes\n\n")





if __name__ == "__main__":

    df = pd.read_parquet(INPUT_FILE)
    # total = len(df)
    # print(f"total number of paragraphs/texts/rows in the input file: {total}")

    # df = df.head(20000) ## TESTING ON FIRST 20,000

    l = len(df)
    print(f"loaded the input file; total number of paragraphs/texts/rows: {l}\n")

    tokenizer, model = load_model()
    print("loaded model and tokenizer\n\n")

    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE, device=0 if torch.cuda.is_available() else -1)


    starting_row = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            starting_row = sum(1 for line in f) - 1
            print(f"found checkpoint file; continue from row {starting_row}\n\n")
    else:
        print("no checkpoint file found; starting from the beginning\n\n")
        with open(CHECKPOINT_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_index", "clf_label", "posterior_prob_score"])


    if starting_row >= len(df):
        print("no more rows to process; complete\n\n")

    else:
        ## changed to process in chunks to prevent OOM error
        # going to process in chunks of 50,000 rows at a time
        rows_remaining = len(df) - starting_row
        for chunk_start in range(starting_row, len(df), CHUNK_SIZE):
            chunk = df.iloc[chunk_start : chunk_start + CHUNK_SIZE].copy()

            print(f"\n\nprocessing chunk rows {chunk_start} to {chunk_start + len(chunk)} (total rows remaining: {len(df) - chunk_start})\n\n")

            # process_batch(chunk, tokenizer, model, starting_row=chunk_start)
            process_batch(chunk, pipe, starting_row=chunk_start)

            ##
            del chunk
            gc.collect()
            print(f"\nmemory cleaned; starting next chunk\n\n")

    print("\n\ndone classifing, now joining results with og df")
    temp_df = pd.read_csv(CHECKPOINT_FILE)

    og_index = set(range(len(df)))
    temp_index = set(temp_df["row_index"])

    if og_index != temp_index:
        missing_rows = og_index - temp_index
        duplicate_rows = temp_df["row_index"].duplicated().sum()
        print(f"number of rows NOT the same between og df nd temp df: {len(missing_rows)}")
        if missing_rows:
            print(f"missing rows: {missing_rows}")  
        if duplicate_rows:
            print(f"number of duplicate rows in temp df: {duplicate_rows}")
        exit()

    else:
        temp_df = temp_df.sort_values("row_index").reset_index(drop=True)

        df["clf_label"] = temp_df["clf_label"].values
        df["posterior_prob_score"] = temp_df["posterior_prob_score"].values



    print("\n\nresulting df after clf:")
    print(df.shape)
    print(df["clf_label"].value_counts())

    climate_related_df = df[df["clf_label"] == "yes"].copy()
    print(f"\ntotal num of climate-related texT: {len(climate_related_df)}\n\n")
    

    climate_related_df.to_parquet(OUTPUT_FILE, index=False)

    print("\n\nSAVED  CLIMATE-RELATED PARAGRAPHS/TEXTS/ROWS TO PARQUET FILE")



