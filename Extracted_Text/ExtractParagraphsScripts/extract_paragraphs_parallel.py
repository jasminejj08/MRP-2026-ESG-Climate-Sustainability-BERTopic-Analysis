# TO UPLOAD

## using extract_paragraphs_PARQUETver.py to extract paragraphs from the pdf report files

# going to parallelize the process; using multiple cpu cores
## --> bc takes a long time to process all pdf files

# relevant documentation: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor


import pandas as pd
import os
import time
## high-level interface for asynchronoulsy executing callables
from concurrent.futures import ProcessPoolExecutor, as_completed

from extract_paragraphs_PARQUETver import check_exploded_df_csv, process_single_report

REPORTS_FOLDER = "/.../Sampled_Files/all_SR_reports/"
EXPLODED_DF_FILE = "/.../MRP/Sampled_Files/final_df.csv"
OUTPUT_PARQUET_FILE = "/.../Sampled_Files/Extracted_Paragraphs_Final/exploded_paragraphs.parquet"

## gonna save after every 1000 reports processed
CHECKPOINT = 1000
# # to speed up as much as possible
NUM_CORES = os.cpu_count()

# first in first served
## function that each core will run to process a report
## ProcessPoolExecutor needs this to serialize 
def each_core(task):

    report_name, row_dict = task

    ## the extract_paragraphs_PARQUETver.py function accepts arguemtns:
    # report_name, source_directory, metadata_row 
    result_rows = process_single_report(report_name, REPORTS_FOLDER, row_dict)

    return report_name, result_rows



def execute_parallel_ext(exploded_df):
    os.makedirs(os.path.dirname(OUTPUT_PARQUET_FILE), exist_ok=True)

    total = len(exploded_df)

    print(f"{NUM_CORES} CPU cores beng used for prallel processing")
    print(f"processing {total} reports\n")

    ## each core will process a report and extract paragraphs from it
    # each core needs report name + metadata row 

    required_info = [

        (row["report_name"], row.to_dict()) 

        for _, row in exploded_df.iterrows()
    ]

    all_rows = []
    missing_reports = []
    completed = 0


    start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_CORES) as executor:
        futures = []

        for task in required_info:
            future = executor.submit(each_core, task)
            futures.append(future)

        ## as_completed() returns an iterator that yields futures as they complete (finished or cancelled)
        # so can process results as they come in; don't let size of report affect speed of processing other reports 
        for future in as_completed(futures):
            report_name, result_rows = future.result()

            completed += 1

            if not result_rows: ## the report not found or processed
                missing_reports.append(report_name)

            all_rows.extend(result_rows) 

            if completed % CHECKPOINT == 0:
                print(f"CHECKPOINT: [{completed}/{total}] reports processed;")
                print(f"reports skipped/missing: {len(missing_reports)}")

                ## save progress so far 
                checkpoint_df = pd.DataFrame(all_rows)
                checkpoint_df.to_parquet(OUTPUT_PARQUET_FILE, index=False)
                print(f"checkpoint SAVED: ({len(checkpoint_df)} paragraphs so far)")

    duration = time.time() - start_time


    print(f"\nprocessing COMPLETED in:\t {duration/60:.2f} minutes")

    print(f"reports skipped/missing:\t {len(missing_reports)}")

    df = pd.DataFrame(all_rows)

    df.to_parquet(OUTPUT_PARQUET_FILE, index=False)
    print(f"\n\nSUCCESSFULLY SAVED {len(df)} paragraphs to:\t {OUTPUT_PARQUET_FILE}")
    print(f"finished in {duration/60:.2f} minutes")

    return df



if __name__ == "__main__":

    exploded_df = check_exploded_df_csv(EXPLODED_DF_FILE)

    print(f"total reports to process: {len(exploded_df)}")
    print("BEGINNING PARALLEL PROCESSING OF REPORTS\n\n")
    
    execute_parallel_ext(exploded_df)
