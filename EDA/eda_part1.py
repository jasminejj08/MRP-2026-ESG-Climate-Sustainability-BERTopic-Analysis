# TO UPLOAD
# Exploratory Data AnalysisS

## this script performs EDA, looking at the distribution of the data
# the report pdf files are looked at to see how many Srs vs ARs there are
## and the associated metadata (firm_id, year, etc.) to determine
# an overview of the data and its distribution

# from this, we can determine how to sample the data for training and testing

# thisl file wiill produce figures and tables that are availalbe in the report

# files used: reports_per_company_year.csv, companies.csv, report_ids.csv
## the first can be found in the processed folder by Forster et al.
# the latter two are in the raw folder

import pandas as pd
import matplotlib.pyplot as plt

# function to load the data and print basic info
def load_data():
    companies = pd.read_csv("companies.csv")
    reports_per_company_year = pd.read_csv("reports_per_company_year.csv")
    report_ids = pd.read_csv("report_ids.csv")

    print("companies.csv shape:", companies.shape)
    print("reports_per_company_year.csv shape:", reports_per_company_year.shape)
    print("report_ids.csv shape:", report_ids.shape)

    return companies, reports_per_company_year, report_ids

# basic unique counts for companies.csv
# Kirstein et al. companies columns: firm, isin, name, country, primary_sics_sector, year
def primary_counts(companies):
    print("\nchecking companies.csv")
    print(companies.head())
    print("\n COUNTS FOR **companies.csv**")
    unique_firms = companies["firm"].nunique()
    unique_countries = companies["country"].nunique()
    unique_sectors = companies["primary_sics_sector"].nunique()
    year_min, year_max = companies["year"].min(), companies["year"].max()

    print(f"\nunique firms: {unique_firms}")
    print(f"unique countries: {unique_countries}")
    print(f"unique sectors: {unique_sectors}")
    print(f"year range: {year_min}-{year_max}")

    ## also checking unique names and ISINs (make sure they match the number of unique firms)
    unique_names = companies["name"].nunique()
    unique_isins = companies["isin"].nunique()

    print(f"unique names: {unique_names}")
    print(f"unique ISINs: {unique_isins}")

    print(unique_names == unique_firms)
    print(unique_isins == unique_firms)

    ## checking for missing values in companies.csv
    print("\nmissing values in companies.csv:\n", companies.isnull().sum())

    #print the rows with missing values in companies.csv
    print("\nrows with missing values in companies.csv:\n", companies[companies.isnull().any(axis=1)])
    ## NOTE THAT THERE ARE 20 missing entries for primary_sics_sector
    ## explained in the report; left like this becaus Kirstin et al. did not classify these firms

    print(f"\nnumber of unique sectors: {companies['primary_sics_sector'].nunique()}")

    print(f"\nsectors: {companies['primary_sics_sector'].unique()}")


    # # figure 1: distribution of UNIQUE firms by country
    # # country on x-axis, number of firms on y-axis
    # firm_counts_by_country = companies.groupby("country")["firm"].nunique().sort_values(ascending=False)

    # plt.figure(figsize=(10, 6))
    # firm_counts_by_country.plot(kind="bar")
    # plt.title("Figure 2.5.A: Distribution of Firms by Country")
    # plt.xlabel("Country")
    # plt.ylabel("Number of Unique Firms")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig("figures/figure_2_5_A_distribution_of_firms_by_country.png")
    # plt.close()

    # unique firms by sector
    # x_axis = primary_sics_sector, y_axis = number of unique firms
    # firm_counts_by_sector = companies.groupby("primary_sics_sector")["firm"].nunique().sort_values(ascending=False)

    # plt.figure(figsize=(10, 6))
    # firm_counts_by_sector.plot(kind="bar")
    # plt.title("Figure 2.5.C: Distribution of Firms by Sector")
    # plt.xlabel("Sector")
    # plt.ylabel("Number of Unique Firms")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig("figures/figure_2_5_C_distribution_of_firms_by_sector.png")
    # plt.close()


# structure of report_ids.csv: report_id
## note that the report_id has the format: firm_id_year_report_type
def report_ids_counts(report_ids):
    print("\nchecking report_ids.csv ------------------------------------------------------")
    print(report_ids.head())

    ## parse the report_id to extract firm_id, year, and report_type
    report_ids[["firm_id", "year", "report_type"]] = report_ids["report_id"].str.rsplit("_", n=2, expand=True)

    print("\n COUNTS FOR **report_ids.csv**")
    unique_report_ids = report_ids["report_id"].nunique()
    unique_firms = report_ids["firm_id"].nunique()
    unique_years = report_ids["year"].nunique()

    print(f"\nunique report IDs: {unique_report_ids}")
    print(f"unique firms: {unique_firms}")
    print(f"unique years: {unique_years}")

    print(f"example unique report id: {report_ids['report_id'].iloc[0]}")
    print(f"example firm_id: {report_ids['firm_id'].iloc[0]}")
    print(f"example report_type: {report_ids['report_type'].iloc[0]}")

    # count number of reports per report_type
    report_type_counts = report_ids["report_type"].value_counts()
    print("report type counts:\n", report_type_counts)

    # there is a single report_type that is "Sr" instead of "SR" --> fixing that
    report_ids["report_type"] = report_ids["report_type"].str.upper()
    print("\nreport type counts after fixing 'Sr' to 'SR':\n", report_ids["report_type"].value_counts())

    ## checking missing values in report_ids.csv
    print("\nmissing values in report_ids.csv:\n", report_ids.isnull().sum())





    ## figure 2: distribution of report types PER YEAR
    ## x-axis: year, y-axis: number of reports
    # report_type_year_counts = report_ids.groupby(["year", "report_type"]).size().unstack(fill_value=0)

    # plt.figure(figsize=(10, 6))
    # report_type_year_counts.plot(kind="bar", stacked=True)
    # plt.title("Figure 2.5.B: Distribution of Report Types per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Number of Reports")
    # plt.legend(title="Report Type")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.savefig("figures/figure_2_5_B_distribution_of_report_types_per_year.png")
    # plt.close()

    # get format company_year (firmid_year) to compare to reports_per_company_year.csv
    report_ids["company_year"] = report_ids["firm_id"] + "_" + report_ids["year"]
    # get unique company_years from report_ids.csv
    unique_company_years_report_ids = report_ids["company_year"].nunique()
    print(f"\nunique company_year in report_ids.csv: {unique_company_years_report_ids}")

    print(f"\nheaders of report_ids.csv: {report_ids.columns.tolist()}")


    # FIGURE: total reports per year (all report types)
    # x-axis = year, y-axis = number of reports --> line plot
    # plt.figure(figsize=(10, 6))
    # report_counts_per_year = report_ids.groupby("year").size()
    # report_counts_per_year.plot(kind="line", marker="o")
    # plt.title("Figure 2.5.D: Total Reports per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Number of Reports")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.grid(True)
    # plt.savefig("figures/figure_2_5_D_total_reports_per_year.png")
    # plt.close()


    # # FIGURE: distribution of report types per year (bar chart with bars side by side)
    # plt.figure(figsize=(10, 6))
    # report_type_year_counts = report_ids.groupby(["year", "report_type"]).size().unstack(fill_value=0)
    # report_type_year_counts.plot(kind="bar", stacked=False)
    # plt.title("Figure 2.5.E: Distribution of Report Types per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Number of Reports")
    # plt.xticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.legend(title="Report Type")
    # plt.grid(True)
    # plt.savefig("figures/figure_2_5_E_distribution_of_report_types_per_year.png")
    # plt.close()
    

# for reports_per_company_year.csv, we can check the number of unique company_years and compare to report_ids.csv
## this is soo that we can see if there are any company_years in the reports_per_company_year.csv that are not in report_ids.csv
## if this is true, then we can just use reports_per_company_year.csv because it has the assocaited reports 
## for each company_year
def reports_per_company_year_counts(reports_per_company_year, report_ids):
    print("\n\nchecking reports_per_company_year.csv ------------------------------------------------------")
    print(reports_per_company_year.head())

    ## get unique company_years from reports_per_company_year.csv
    unique_company_years_reports_per_company_year = reports_per_company_year["company_year"].nunique()
    print(f"\nunique company_year in reports_per_company_year.csv: {unique_company_years_reports_per_company_year}")

    ## get unique company_years from report_ids.csv
    unique_company_years_report_ids = report_ids["company_year"].nunique()
    print(f"\nunique company_year in report_ids.csv: {unique_company_years_report_ids}")

    ## check if there are any company_years in reports_per_company_year.csv that are not in report_ids.csv
    missing_company_years = set(reports_per_company_year["company_year"]) - set(report_ids["company_year"])
    print(f"\nnumber of company_years in reports_per_company_year.csv that are not in report_ids.csv: {len(missing_company_years)}")
    if len(missing_company_years) > 0:
        print("missing company_years:", missing_company_years)
    else:
        print("no missing company_years")

    
    ## also check if the number of reports in the reports column of reports_per_company_year.csv file
    ## matches the number of pdf report files in the other zip folder (the one with the actual pdf files)
    ## this is to check that the reports_per_company_year.csv file is accurate and can be used instead of report_ids.csv
    reports_per_company_year["num_reports"] = reports_per_company_year["reports"].apply(lambda x: len(x.split(",")) if pd.notnull(x) and x.strip() != "[]" else 0)

    print(f"\ntotal number of reports in reports_per_company_year.csv (from 'reports' column): {reports_per_company_year['num_reports'].sum()}")

    ## the total number of reports in reports_per_company_year.csv matches the total number of pdf report files in [ADDRESS HERE]
    ## --> can use reports_per_company_year.csv instead of report_ids.csv because it has the associated reports for each company_year

    ## check missing values in reports_per_company_year.csv
    print("\nmissing values in reports_per_company_year.csv:\n", reports_per_company_year.isnull().sum())




## figures that require data to be merged/use more than one file in separate function

def merged_data_counts(companies, reports_per_company_year, report_ids):
    print("\n\nCURRENT METHOD: merged_data_counts() ------------------------------------------------------")
    print("\nWORKING WITH MERGED DATA")

    print(f"length of companies.csv: {len(companies)}")
    print(f"length of reports_per_company_year.csv: {len(reports_per_company_year)}")

    ## want to take a look at how the sector relates to the type of reports per year
    ## this will require merging the companies.csv and reports_per_company_year.csv 
    ## both can be merged on firm_id --> need to create firm_id column in reports_per_company_year.csv by splitting 

    ## first create company_year column in companies.csv by combining firm and year columns
    companies["company_year"] = companies["firm"].astype(str) + "_" + companies["year"].astype(str)
    # print("\nchecking companies.csv after adding company_year column")
    # print(companies.head())
    # print first row of companies.csv after adding company_year column
    # print(companies.iloc[0])

    ## join companies.csv and reports_per_company_year.csv on company_year column
    ## keep all rows from companies.csv and add reports_per_company_year.csv data where available (left join)
    merged_df = pd.merge(companies, reports_per_company_year, on="company_year", how="left")

    # print("\nchecking merged data")
    # print(merged_df.head())

    # # first two rows of merged_df after merging companies.csv and reports_per_company_year.csv
    # print(merged_df.iloc[0])
    # print(merged_df.iloc[1])

    # print(f"headers of merged_df: {merged_df.columns.tolist()}")

    # ### check the reports column of merged_df to see if it contains the report_ids from report_ids.csv
    # print(merged_df["reports"].iloc[0])

    ## check number of reports --> should match total number of reports in report_ids.csv / the total number of pdf report files in the zip folder 
    print(f"total number of reports in merged df: {merged_df['num_reports'].sum()}")
    ## MATCHES :D

    print(f"length of merged df: {len(merged_df)}")

    # ## saving this as a csv file to open in excel
    # merged_df.to_csv("merged_companies_reports_per_company_year.csv", index=False)
    # ## SAVED --> going to upload this to github as well

    ## check for missing values in the merged dataframe
    missing_values = merged_df.isnull().sum()
    print("\nmissing values in merged dataframe:\n", missing_values)

    ## print the rows with missing values in the merged dataframe
    print("\nrows with missing values in merged dataframe:\n", merged_df[merged_df.isnull().any(axis=1)])

    ## any company-years with no reports (num_reports = 0) check
    print("\ncompany-years with no (null) reports:\n")
    print(merged_df[merged_df["num_reports"].isnull()][["firm", "year", "company_year", "num_reports"]])

    missing_sector_rows = companies[companies["primary_sics_sector"].isnull()]
    print(missing_sector_rows[["firm", "name", "country", "year"]].drop_duplicates(subset="firm"))

    # investigate missing reports on their own
    missing_report_rows = merged_df[merged_df["reports"].isnull()]
    print(missing_report_rows.groupby("year").size())
    print(missing_report_rows.groupby("country").size())

    firm_report_totals = merged_df.groupby("firm")["num_reports"].sum()
    firms_with_zero_total = firm_report_totals[firm_report_totals == 0]
    print(f"firms with zero reports across ALL years: {len(firms_with_zero_total)}")


    ## GO TO  PART 2 OF THE EDA: using the merged data to zoom into sectors and report types for stratified sampling

if __name__ == "__main__":
    companies, reports_per_company_year, report_ids = load_data()

    primary_counts(companies)
    report_ids_counts(report_ids)
    reports_per_company_year_counts(reports_per_company_year, report_ids)

    merged_data_counts(companies, reports_per_company_year, report_ids)
