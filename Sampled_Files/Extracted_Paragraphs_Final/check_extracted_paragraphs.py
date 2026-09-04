# TO UPLOAD 

## checking the exploded_paragraphs.parquet file as checkup before
# applying text classification (climteBERT)


import pandas as pd

OUTPUT_FILE = "exploded_paragraphs.parquet"

def check_extracted_paragraphs(output_paragraphs_df):
    # print(output_paragraphs_df.head())
    print(f"num of paragraphs: {len(output_paragraphs_df)}")
    print(f"shape: {output_paragraphs_df.shape}")
    # print(f"columns: {output_paragraphs_df.columns}")
    # print(f"missing values: {output_paragraphs_df.isnull().sum()}")

    # print(output_paragraphs_df["year"].value_counts().sort_index())
    # print("\n", output_paragraphs_df["primary_sics_sector"].value_counts())
    print("\n", output_paragraphs_df["word_count"].describe())
    # print("\n", output_paragraphs_df["text"].sample(5).to_list())

    ## check missing reports (55)
    unique_reports = set(output_paragraphs_df["report_name"].unique())
    print(f"unique reports: {len(unique_reports)}")

    final_df = pd.read_csv("/home/jasmine/masters/deep_learning/MRP/Sampled_Files/final_df.csv")
    final_df_reports = set(final_df["report_name"].unique())

    missing_reports = final_df_reports - unique_reports
    print(f"missing reports from original final_df: {len(missing_reports)}") ## should be 55
    

## RUN CLEAN_EXTRACTED_TEXT.py NEXT
## --> need to drop non-english paragraphs 



if __name__ == "__main__":
    output_paragraphs_df = pd.read_parquet(OUTPUT_FILE)
    check_extracted_paragraphs(output_paragraphs_df)

