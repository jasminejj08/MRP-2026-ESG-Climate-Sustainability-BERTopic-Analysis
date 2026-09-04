## this file is used to stratify the data
## WILL PRODUCE A SEPARATE FILE 
## WILL LOOK AT THE PDF REPORT FILES AND move the relevant SR reports to a separate folder 

## then the SR reports will be ready to run pymupdf on and extract text only from the SR reports
# andd that file will remove any non-usable reports; chunk into paragraphs; create new csv file (ataseT)
## refer to the methodology section of the report for more details on this process

# using the merged_df 

## look at main to see code flow + set sampling = true or false
# default is using all the sr reports for each year (excluding missing sector rows)

import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

def load_data(file_path):
    df = pd.read_csv(file_path)

    print(f"data loaded shaepe: {df.shape}")

    print(f"columns: {df.columns}")

    print("\n\nfilling in missing values")

    df["primary_sics_sector"] = df["primary_sics_sector"].fillna("")

    df["reports"] = df["reports"].fillna("") # no reports
    df["num_reports"] = df["num_reports"].fillna(0)


    # print("\nmissing values in df after filling in missing values:")
    # print(df.isnull().sum())


    df["num_SR_reports"] = df["reports"].apply(
        lambda x: sum(1 for report in x.split(",") if report.strip().strip("[]'\" ").upper().endswith("_SR.PDF"))
    )
    df["num_AR_reports"] = df["reports"].apply(
        lambda x: sum(1 for report in x.split(",") if report.strip().strip("[]'\" ").upper().endswith("_AR.PDF"))
    )

    print(f"\n\ncolumns after: {df.columns}")

    print(f"\ntotal nubmer of SR reports vs AR reports in merged_df:")
    print(df[["num_SR_reports", "num_AR_reports"]].sum())

    return df


# Stratify the data based on primary_sics_sector and num_SR_reports
## idea: take the same amount of SR reports from each sector for every year
# --> working with percentages of SR reports per sector per year; then can sample the same percentage of SR reports from each sector for every year

## want to select 100 (initial) SR reports for each year; such that the sector mix of sample matches the sector mix of the population of SR reports for that year
## using PROPORTIONAL STRATIFIED SAMPLING (random); --> sample the same proportion of SR reports from each sector for every year (to stay consistent with the population of SR reports for that year)

# need to 
# for proportional, need to get the percentage of SR reports for each sector for each year
# first count number of SR reports for each sector for each year --> need to explode reports column (list of strings) in separate reows to filer SR olny


def explode_reports_column(df):
    # amking ever y report in reports column a separate row 

    print(f"\n\ncolumns of current df: {df.columns}")


    df = df[(df["primary_sics_sector"] != "") & (df["num_SR_reports"] > 0)] # filter out rows with missing primary_sics_sector and rows with no SR reports

    print(f"\n\nshape of df after filtering out rows with missing primary_sics_sector and rows with no SR reports: {df.shape}")
    # print(f"\n\ncolumns of df after filtering out rows with missing primary_sics_sector and rows with no SR reports: {df.columns}")
    print(f"\n\nhead of df after filtering out rows with missing primary_sics_sector and rows with no SR reports: {df.head()}")

    print(f"\n\ncolumns of filtered df: {df.columns}")

    ## SELF-NOTE:
    ## so df currently has rows that have at least one SR report and a primary_sics_sector value; 
    ## --> can also have AR reports, but any columns with missing primary_sics_sector or no SR reports have been filtered out ==> 
    ## even if a row had an AR report but no SR report its filtered out
    ## went from 3477 SR reports to 3473; meaning 5 rows had no primary_sics seector value or no SR reports;



    df_exploded = df.copy()
    df_exploded['report_name'] = df_exploded['reports'].apply(
        lambda x: [r.strip().strip("[]'\" ") for r in x.split(",")] if x else []
    )

    df_exploded = df_exploded.explode('report_name')

    ## for SR only reports
    df_exploded = df_exploded[df_exploded['report_name'].str.upper().str.endswith('_SR.PDF')]
    
    checking = df_exploded.groupby(["primary_sics_sector", "year"]).size().unstack(fill_value=0)
    print("\n\nnumber of SR reports per sector per year:\n", checking) ## like a matrix where x = years y = sectors and values = number of SR reports))
    print("\ncolumns of checking df (years):\n", checking.columns)

    ## checking the exploded dataframe 
    print("\n\nexploded df shape: ", df_exploded.shape)
    print("\n\nexploded df columns: ", df_exploded.columns)
    print("\n\nexploded df head: ", df_exploded.head())

    print("\n\nexploded df more info: ", df_exploded.info())
    print("\n\nexploded df head: ", df_exploded[['report_name','primary_sics_sector','year']].head())


    yearly_counts = checking.sum(axis=0)
    print("\n\nnumber of SR reports per year:\n", yearly_counts)

    ## going by year to get sector % of SR reports for each year
    percent = checking.div(yearly_counts, axis=1)
    print("\n\npercentage of SR reports per sector per year :\n", percent)
    # percent will also indicate % of SR reports to smaple for each sector for each year 



    return df_exploded, checking, percent


## currently just have the percentages of SR reports per sector per year
## --> essentially kind of maping them to the actual number of reports to sample for each sector for each year
# based on the % found in percent df
def percent_to_sample(checking, percent, sample_size=250):
    ## transform percentages into number of reports to sample for each sector for each year 

    ## if the number is less than 1, round up to 1 ( ensure at least one report is sampled from each sector for each year)
    sample_counts = (percent * sample_size).round().astype(int)

    round_true = (checking > 0) & (sample_counts < 1) 
    sample_counts[round_true] = 1

    ## check that not exceeding number of reports available for each sector for each year
    sample_counts = sample_counts.clip(upper=checking)

    print("\n\nnumber of reports to sample for each sector for each year:\n", sample_counts)

    return sample_counts


## now need to get the actual sample of reports based on the sample_counts df (the report file names)
# --> will copy the sampled reports to a separate folder
## ACTUAL SAMPLING PART HERE
# 
def sample_reports(df_exploded, sample_counts, random_state=42):
    ## store sampled rows in list;
    sampled_rows = []

    ## sample_counts is a df with sectors as index (rows) and years as columns; values = number of reports to sample for that sector,year 
    for sector in sample_counts.index:
        for year in sample_counts.columns:
            x = sample_counts.loc[sector, year]

            if x == 0:
                continue ## no reports for this sector,year; skip


            sampled = df_exploded[(df_exploded['primary_sics_sector'] == sector) & (df_exploded['year'] == year)].sample(n=x, random_state=random_state)
            sampled_rows.append(sampled)

    ## concatenate all sampled rows into a single dataframe
    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    print(f"\n\nsampled_df shape: {sampled_df.shape}") # rows = number of sampled roports, columns = original columns + report_name column
    print(f"\n\ntotal number of sampled sr reports: {len(sampled_df)}")

    print(f"\n\ntotal number of sampled sr reports: {sampled_df.shape[0]}")


    return sampled_df




## for the sake of BERTopic, using ALL the SR reports for each year (excluding missing sector rows)
## --> good to get more coherent topics

# function to copy all SR reports to a separate folder 
def copy_reports(df, source_directory, destination_dir):
    
    os.makedirs(destination_dir, exist_ok=True)

    copied_reports, skipped, missing_reports = 0, 0, []

    total_reports = len(df)

    for report_name in df['report_name']:
        src_path = os.path.join(source_directory, report_name)

        ##
        cleaned_report_name = report_name.replace("_Sr.pdf", "_SR.pdf").replace("_sr.pdf", "_SR.pdf") ## for that one report that had a lower case from the EDA

        dst_path = os.path.join(destination_dir, cleaned_report_name)


        if os.path.exists(src_path):
            ##
            shutil.copy(src_path, dst_path)


            ##
            copied_reports += 1

            print(f"copying progress: {copied_reports}/{total_reports}\n")
        else:
            skipped += 1
            print(f"report {report_name} not found; SKIPPED; copying progress: {copied_reports}/{total_reports}\n")
            missing_reports.append(report_name)

    print(f"\n\nCopied {copied_reports} reports to {destination_dir}")
    print(f"Skipped {skipped} reports ")
    if missing_reports:
        print(f"MISSING REPORTS: {len(missing_reports)} reports; overview: {missing_reports[:5]}")

    return copied_reports, missing_reports


if __name__ == "__main__":
    ## MAY NEED TO adjust path to ur directory
    working_df = load_data("merged_companies_reports_per_company_year.csv")

    exploded_df, checking, percent = explode_reports_column(working_df)

    ## going to use ALL SR reports for my case; but can change to smapling
    ## globsl variable
    USE_PERCENTAGE_SAMPLING = False 

    ## ma y need to change the folder path to ur directory if running on ur own machine
    ## 
    if USE_PERCENTAGE_SAMPLING:
        sample_counts = percent_to_sample(checking, percent, sample_size=100)
        final_df = sample_reports(exploded_df, sample_counts, random_state=42)
        folder = "/home/jasmine/masters/deep_learning/MRP/Sampled_Files/sampled_SR_reports"
    else:
        final_df = exploded_df.copy() ## using all SR reports for each year (excluding missing sector rows)
        folder = "/home/jasmine/masters/deep_learning/MRP/Sampled_Files/all_SR_reports"

    # save final_df as csv to folder directory
    copy_reports(final_df, source_directory="/home/jasmine/masters/deep_learning/MRP/pdf_reports/reports_pdf_extracted/reports_pdf_final_run_2_final", destination_dir=folder) ## need to adjust path to ur directory
    final_df.to_csv("/home/jasmine/masters/deep_learning/MRP/Sampled_Files/final_df.csv", index=False)


