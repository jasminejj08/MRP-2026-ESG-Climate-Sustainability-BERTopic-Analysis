# TO UPLOAD

## noticed some paragraphs aren't in English in the extracted paragraphs
# --> going to drop these paragraphs (keeping ENGLISH-ONLY paragraphs)
# going to check if paragraph has mainly English words (using langdetect library)

## parallelized again for speedup

import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

## absolute path to files have been removed; change as needed
INPUT_FILE = "/.../Sampled_Files/Extracted_Paragraphs_Final/exploded_paragraphs.parquet"
NUM_CORES = os.cpu_count()


DetectorFactory.seed = 0 

def is_english(text):

    if pd.isna(text):
        return False

    if not isinstance(text, str):
        return False

    text = text.strip()
    if not text:
        return False

    try:
        return detect(text) == "en"
    except LangDetectException:
        ## droping text that can't be detected/fails to be detected
        return False


def process_chunk(texts):
    return [is_english(text) for text in texts]


## divindng the current dataframe into chunks to parallelize
# going to have chunks = num_cores
def chunk_dataframe(l, num_chunks):
    chunk_size = len(l) // num_chunks + 1
    chunks = []

    for i in range(0, len(l), chunk_size):
        chunks.append(l[i:i + chunk_size])

    return chunks


if __name__ == "__main__":
    start = time.time()

    output_paragraphs_df = pd.read_parquet(INPUT_FILE)

    text_list = output_paragraphs_df["text"].tolist()
    text_chunks = chunk_dataframe(text_list, 2000) 

    results = [None] * len(text_chunks)

    ## parallelize language detection using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(text_chunks)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()


    is_english_column = []

    for chunk_result in results:
        for j in chunk_result:
            is_english_column.append(j)

    output_paragraphs_df["is_english"] = is_english_column

    # output_pargraphs_df["is_english"] = output_pargraphs_df["text"].apply(is_english)
    print(output_paragraphs_df["is_english"].value_counts())

    end = time.time()
    total_time = end - start
    print(f"total time taken: {total_time/60:.2f} minutes")

    english_only_df = output_paragraphs_df[output_paragraphs_df["is_english"]].reset_index(drop=True)

    english_only_df.to_parquet("/.../Sampled_Files/Extracted_Paragraphs_Final/english_only_df.parquet", index=False)
    print(f"english_only_df.parquet SUCCESSFULLY SAVED")
