# ver 5 

# TO UPLOAD

####### FINAL VER

## SENTIMENT ANALYSIS using ClimateBert

# input file = cleaned_classified_paragraphs.parquet (generated from clf step)
#

# same idea as the clf but using diff model + added some details bc different labels
## aslso accouting for splitting neutral into pos or neg to make full use of all data based on score lvl

# relevant documentation:
# https://huggingface.co/climatebert/distilroberta-base-climate-sentiment
# https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html
# 


# now have 3 labels: opportunity (positive), risk (negative), neutral
## this sentiment step is to classify the paragraphs into these categories, where
## for the neutral group, the score will determine whether it is more positive or negative
# and hence put into that group --> end up w/ 2 groups (positive and negative)


## most recent change: moved the neutral clf logic to sep file 
# this fille will only do sentiment clf 
# save: checkpoint file for sentiment clf results -_> has row_index, sentiment_label, score 
#    - sentiment_label = positive, negative, neutral
# only save the checkpoint file in main func so after checking that the index matches with og df
# the joining will also be done in the sep file 
# other file will do the neutral clf and join those columns to the og df and save the final df with sentiment_label and score columns

# the og parquet file (from text clf step) will remain unchanged; only creating a new csv file for the sentiment clf results
# the new parquet file will be created in the neutral clf step file


import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import time
from datasets import Dataset
import gc
import csv

# OUTPUT_FILE = "/home/jasmine/masters/deep_learning/MRP/ClimateBERT_SentimentAN/sentiment_clf_paragraphs.parquet"

MODEL_NAME = "climatebert/distilroberta-base-climate-sentiment"
BATCH_SIZE = 128

INPUT_FILE = "/home.../MRP/ClimateBERT_TextCLF/cleaned_classified_paragraphs.parquet"

## since about 300k paragraphs, will print statemnt every 100k paragraphs
BATCH_CHECKPOINT = 100000

CHECKPOINT_FILE = "/home/.../MRP/ClimateBERT_SentimentAN/sentiment_clf_checkpoint_final2.csv"

CHUNK_SIZE = 50000

# NEUTRAL_CLF_FILE = "/home/jasmine/masters/deep_learning/MRP/ClimateBERT_SentimentAN/sentiment_clf_neutral_checkpoint.csv"

d = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    print(model.config.id2label)

    return tokenizer, model


def process_batch(dataset_chunk, pipe, starting_row=0):

    ## changed to pass chunk of Dataset.from_pandas(df) instead of the entire df

    # dataset = Dataset.from_pandas(df)

    ## moved time to main function so that we get total time for entire process; left unchagned for prev step

    print(f"\n\nSTARTING SENTIMENT CLF for {len(dataset_chunk)} paragraphs\n\n")


    with open(CHECKPOINT_FILE, "a", newline="", buffering=1) as f:
        w = csv.writer(f)

        for i, out in enumerate(tqdm(pipe(KeyDataset(dataset_chunk, "text"), padding=True, truncation=True), total=len(dataset_chunk))):

            scores = {item["label"]: item["score"] for item in out}
            w.writerow([i + starting_row, scores.get("opportunity", 0.0), scores.get("neutral", 0.0), scores.get("risk", 0.0), max(scores, key=scores.get)])

            # w.writerow([i + starting_row, out["label"], out["score"]])

            if (i + 1) % BATCH_CHECKPOINT == 0:
                print(f"\nprocessed {i + 1 + starting_row} paragraphs; current run time: {(time.time() - start_time)/60:.2f} minutes\n\n")


    print(f"\nDONE SENTIMENT clf (CLIMATEBERT SENTIMENT) for current batch ")
    print(f"\nbatch run time: {(time.time() - start_time)/60:.2f} minutes\n\n")


## MOVED TO NEW FILE ; this file will do sentiment clf ONLY



if __name__ == "__main__":

    df = pd.read_parquet(INPUT_FILE)

    # df = df.head(20000)

    total = len(df)
    print(f"\n\nloaded the input file; total num of paragraphs: {total}\n")

    

    tokenizer, model = load_model()
    print("\nmodel and tokenizer loaded\n\n")



    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, batch_size=BATCH_SIZE, top_k=None, device=0 if torch.cuda.is_available() else -1)

    starting_row = 0
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            starting_row = sum(1 for line in f) - 1
            print(f"found checkpoint file; continue from row {starting_row}\n\n")
    else:
        print("no checkpoint file found; starting from the beginning\n\n")
        with open(CHECKPOINT_FILE, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row_index", "opportunity_score", "neutral_score", "risk_score", "predicted_label"])
    

    start_time = time.time()

    if starting_row >= len(df):
        print("no more rows to process; complete\n\n")

    else:
        ## processing ramining rows in batches of CHUNK_SIZE

        rows_remaining = len(df) - starting_row

        for chunk_start in range(starting_row, len(df), CHUNK_SIZE):
            chunk = df.iloc[chunk_start : chunk_start + CHUNK_SIZE].copy() ## the current chunk of data to process

            ## moved to pass chunk of Dataset.from_pandas(df) instead of the entire df
            dataset_chunk = Dataset.from_pandas(chunk)

            print(f"\n\nprocessing chunk rows {chunk_start} to {chunk_start + len(chunk)} (total rows remaining: {rows_remaining})\n\n")

            process_batch(dataset_chunk, pipe, starting_row=chunk_start)


            del chunk
            gc.collect()
            print(f"\n memory cleaned; starting next chun k \n\n")

        
        print("\n\ndone sentiment clf for ALL paragraphs;\n\n")

        sentiment_df = pd.read_csv(CHECKPOINT_FILE)

        ## printing statistics of sentiment clf results before doing the neutral clf b
        print("\nsentiment clf results statistics:")
        print(f"\n\nnum of opportunity paragraphs: {len(sentiment_df[sentiment_df['predicted_label'] == 'opportunity'])}")
        print(f"\n\nnum of risk paragraphs: {len(sentiment_df[sentiment_df['predicted_label'] == 'risk'])}")
        print(f"\n\nnum of neutral paragraphs: {len(sentiment_df[sentiment_df['predicted_label'] == 'neutral'])}")


        ## check if index of sentiment_df amtches original df (make sure no rows were skipped or duplicated)
        # if all good, print success message and continue to neutral clf file
        og_i = set(range(len(df)))
        sentiment_i = set(sentiment_df["row_index"])

        if sentiment_i == og_i:
            print("\n\nsentiment clf results index matches original df index; ALL GOOD: MOVE TO NEXT FILE\n\n")

        else:
            print("\n\nsentiment clf results index (in checkpoint file) does NOT match og df index; CHECK FOR MISSING OR DUPLICATE ROWS\n\n")
            missing_rows = og_i - sentiment_i
            print(f"\nnumber of missing rows in sentiment clf df: {len(missing_rows)}")
            duplicate_rows = sentiment_df["row_index"].duplicated().sum()
            if duplicate_rows:
                print(f"number of duplicate rows in sentiment clf df: {duplicate_rows}")


        print("\n\nscript complete")
        print(f"\n\nSAVED SENTIMENT CLF RESULTS TO CHECKPOINT FILE: {CHECKPOINT_FILE}\n\n")



## files produce:
# - checkpoint file for sentiment clf results (row_index, sentiment_label, score) : sentiment_clf_checkpoint.csv ## OLD VER
# - new file has: (row_index, opportunity_score, neutral_score, risk_score, predicted_label) : sentiment_clf_checkpoint_final2.csv

# files used:
# - input file: cleaned_classified_paragraphs.parquet (from text clf step)
