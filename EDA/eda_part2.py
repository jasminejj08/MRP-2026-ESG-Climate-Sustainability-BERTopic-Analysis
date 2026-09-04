## PART 2 OF THE EDA: focusing on SR reports only

## generating some figures and tables for SR reports only
## this is to determine how to sample the data (stratified)

## this file uses the merged_companies_reports_per_company_year.csv file generated in eda_part1.py
# run eda_part1.py first to generate the merged_companies_reports_per_company_year.csv file

## from eda_part1.py, we have the following figures created:
# - distribution of firms by country
# - distribution of report types per year (two graphs, one stacked bar, one side by side bar)
# - distribution of firms by sector
# - total number of reports per year


import pandas as pd
import matplotlib.pyplot as plt

def load_merged_data():
    merged_df = pd.read_csv("merged_companies_reports_per_company_year.csv")

    print("merged_df shape:", merged_df.shape)
    print("merged_df columns:", merged_df.columns)

    print(f"\ntotal number of reports in merged_df: {merged_df['num_reports'].sum()}")
    
    return merged_df


## going to generate some figures and tables
def generate_analysis(merged_df):
    ## before focusing on one particular type of report, want to see connection between
    ## report types and sectors, countries, and years

    print("missing values in merged_df:")
    print(merged_df.isnull().sum())

    ## rows with missing values
    print("\nrows with missing values in merged_df:")
    print(merged_df[merged_df.isnull().any(axis=1)])

    ## first fillng in "" for missing values in report_type, sector, and reports column
    ## fill in 0 for missing values in num_reports column
    merged_df["primary_sics_sector"] = merged_df["primary_sics_sector"].fillna("")


    merged_df["reports"] = merged_df["reports"].fillna("") # no reports
    merged_df["num_reports"] = merged_df["num_reports"].fillna(0)



    print("\nmissing values in merged_df after filling in missing values:")
    print(merged_df.isnull().sum())

    print("\nnumber of total reports in merged_df:", merged_df["num_reports"].sum())
    print("total number of unique countries in merged_df:", merged_df["country"].nunique())


    ## FIGURE: number of reports per country 
    # x-axis: country, y-axis: number of reports, horizontal bar chart
    # plt.figure(figsize=(10, 6))
    # reports_per_country = merged_df.groupby("country")["num_reports"].sum()
    # reports_per_country.plot(kind="barh")
    # plt.title("Number of Reports per Country")
    # plt.xlabel("Number of Reports")
    # plt.ylabel("Country")
    # plt.tight_layout()
    # plt.grid(axis="x")
    # plt.savefig("figures_part2/reports_per_country.png")
    # plt.close()


    ## FIGURE: distribution of report type per country (horizontal bar chart with bars side by side)
    # need bigger figure size for this one
    ## need to count number of SR reports vs country and AR reports vs country --> currently reports column has both SR and AR reports in a list
    # (a list of reports (firm_year_reporttype) for each company_year, so need to count number of SR reports and AR reports for each country
    ## adding in two new columns to merged_df: num_SR_reports and num_AR_reports
    ## need to split the reports column / go through each element in the list in reports column 
    ## and count if that element contains "SR" or "AR" and sum them up for each row
    # merged_df["num_SR_reports"] = merged_df["reports"].apply(lambda x: sum(1 for report in x.split(",") if "SR" in report or "Sr" in report or "sr" in report or "sR" in report))
    # merged_df["num_AR_reports"] = merged_df["reports"].apply(lambda x: sum(1 for report in x.split(",") if "AR" in report or "Ar" in report or "ar" in report or "aR" in report))
    ## needed to account for that one row that had "Sr" instead of SR (note from eda_part1)
    merged_df["num_SR_reports"] = merged_df["reports"].apply(
    lambda x: sum(1 for report in x.split(",") if report.strip().strip("[]'\" ").upper().endswith("_SR.PDF"))
    )
    merged_df["num_AR_reports"] = merged_df["reports"].apply(
        lambda x: sum(1 for report in x.split(",") if report.strip().strip("[]'\" ").upper().endswith("_AR.PDF"))
    )

    ## --> for the "Sr" issue
    ## ceck to to see if num_SR_reports + num_AR_reports = num_reports for each row
    merged_df["num_reports_check"] = merged_df["num_SR_reports"] + merged_df["num_AR_reports"]
    if (merged_df["num_reports_check"] != merged_df["num_reports"]).any():
        print("Warning: num_SR_reports + num_AR_reports != num_reports for some rows in merged_df")
    else:
        print("num_SR_reports + num_AR_reports = num_reports for all rows in merged_df")

    # prin the rows where num_SR_reports + num_AR_reports != num_reports
    print(merged_df[merged_df["num_reports_check"] != merged_df["num_reports"]])
    ######-------------------

    # ## SAVING AS NEW CSV FILE (for later use in sampling.py) --> need number of SR reports and AR reports columns
    # try:
    #     merged_df.to_csv("merged_main.csv", index=False)
    #     print("merged_main.csv saved successfully.")
    # except Exception as e:
    #     print(f"Error saving merged_main.csv: {e}")



    ## now can plot distribution of report type per country (horizontal bar chart with bars side by side)
    ## keeping AR as blue and SR as orage
    # plt.figure(figsize=(12, 8))
    # reports_per_country = merged_df.groupby("country")[["num_SR_reports", "num_AR_reports"]].sum()
    # reports_per_country.plot(kind="barh", stacked=False, color=["orange", "tab:blue"])
    # plt.title("Distribution of Report Type per Country")
    # plt.xlabel("Number of Reports")
    # plt.ylabel("Country")
    # plt.tight_layout()
    # plt.grid(axis="x")
    # plt.savefig("figures_part2/distribution_report_type_per_country.png")
    # plt.close()


## there were some sectors not filled in, going to do sector-specif c analysis in a sep func
## going to first check the names of the firms missing sectors and check
## which one of them have reports --> check the report types for these; (pre-analysis)

def sector_specific_analysis(merged_df):

    print("\n total number of SR reports vs AR reports in merged_df:")
    print(merged_df[["num_SR_reports", "num_AR_reports"]].sum())
    
    missing_sector_names = merged_df[merged_df["primary_sics_sector"] == ""]["name"].unique()
    row_no_sector_has_reports = merged_df[(merged_df["primary_sics_sector"] == "") & (merged_df["num_reports"] > 0)]

    print(f"\ntotal number of reports (1) : {merged_df['num_reports'].sum()}")

    print(f"\nnames of firms missing sectors: {missing_sector_names}")
    print(f"\nnumber of rows with missing sectors but have reports: {len(row_no_sector_has_reports)}")
    print(f"\nrows with missing sectors but have reports: {row_no_sector_has_reports}")

    ## df with rows with missing sectors but have reports is row_no_sector_has_reports
    ## use this to check the number of report types for these firms
    print(f"\nreport types for firms with missing sectors but have reports: {row_no_sector_has_reports[['name', 'num_SR_reports', 'num_AR_reports']]}")


    ## there are 7 rows with missing sectors but hahve reports, these are froma Allegro and Wise;
    ## with Allegro having few SR reports and Wise having none
    ## --> going to DROP all rows with misisng sectors for sector-speific analyis
    ## Aleergo and Wise are excluded from sector-based analysis

    merged_df_sector = merged_df[merged_df["primary_sics_sector"] != ""] ## used for sector specific figures
    print(f"\ntotal number of reports in merged_df_sector: {merged_df_sector['num_reports'].sum()}") ## number of remaining reports after dropping rows with missing sectors

    # ## FIGURE: distribution of reports per sector (bar chart with bars side by side)
    # plt.figure(figsize=(12, 8))
    # reports_per_sector = merged_df_sector.groupby("primary_sics_sector")[["num_SR_reports", "num_AR_reports"]].sum()
    # reports_per_sector.plot(kind="barh", stacked=False, color=["orange", "tab:blue"])
    # plt.title("Distribution of Report Type per Sector")
    # plt.xlabel("Number of Reports")
    # plt.ylabel("Sector")
    # plt.yticks(rotation=45, ha="right")
    # plt.tight_layout()
    # plt.grid(axis="x")
    # plt.savefig("figures_part2/distribution_report_type_per_sector_v3.png")
    # plt.close()

    ## FIGURE: distribution of reports per sector per year
    ## since there are 10 years, will use heatmap 
    ## --> darker color means more reports for that sector in that year --> going to make one for SR reports and another for AR reports
    ## for SR reports
    # SR_sector_year = merged_df_sector.groupby(["primary_sics_sector", "year"])["num_SR_reports"].sum().unstack(fill_value=0)
    # AR_sector_year = merged_df_sector.groupby(["primary_sics_sector", "year"])["num_AR_reports"].sum().unstack(fill_value=0)

    # ## overall heatmap for all SR reports per sector per year
    # plt.figure(figsize=(12, 8))
    # plt.imshow(SR_sector_year, cmap="Oranges", aspect="auto")
    # plt.colorbar(label="Number of SR Reports")
    # plt.xticks(range(len(SR_sector_year.columns)), SR_sector_year.columns, rotation=45)
    # plt.yticks(range(len(SR_sector_year.index)), SR_sector_year.index)
    # plt.title("Distribution of SR Reports per Sector per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Sector")
    # plt.tight_layout()
    # plt.savefig("figures_part2/SR_sector_year_heatmap_v3.png")
    # plt.close()

    # # ## too determin % of SR reports per sector per year,, heatmap of % of SR reports per sector per year
    # # SR_sector_year_percentage = SR_sector_year.div(SR_sector_year.sum(axis=0), axis=1) * 100

    # ## --> UPDATED
    # year_totals = SR_sector_year.sum(axis=0)
    # SR_sector_year_percentage = (SR_sector_year.div(year_totals, axis=1) * 100).fillna(0) 

    # plt.figure(figsize=(12, 8))
    # plt.imshow(SR_sector_year_percentage, cmap="Oranges", aspect="auto")
    # plt.colorbar(label="% of SR Reports")
    # plt.xticks(range(len(SR_sector_year_percentage.columns)), SR_sector_year_percentage.columns, rotation=45)
    # plt.yticks(range(len(SR_sector_year_percentage.index)), SR_sector_year_percentage.index)
    # plt.title("% of SR Reports per Sector per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Sector")
    # plt.tight_layout()
    # plt.savefig("figures_part2/SR_sector_year_percentage_heatmap_v3.png")
    # plt.close()


    # # ## same for AR reports
    # plt.figure(figsize=(12, 8))
    # plt.imshow(AR_sector_year, cmap="Reds", aspect="auto")
    # plt.colorbar(label="Number of AR Reports")
    # plt.xticks(range(len(AR_sector_year.columns)), AR_sector_year.columns, rotation=45)
    # plt.yticks(range(len(AR_sector_year.index)), AR_sector_year.index)
    # plt.title("Distribution of AR Reports per Sector per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Sector")
    # plt.tight_layout()
    # plt.savefig("figures_part2/AR_sector_year_heatmap_v3.png")
    # plt.close()
    
    # # AR_sector_year_percentage = AR_sector_year.div(AR_sector_year.sum(axis=0), axis=1) * 100
    # ## --> UPDATED
    # year_totals_AR = AR_sector_year.sum(axis=0)
    # AR_sector_year_percentage = (AR_sector_year.div(year_totals_AR, axis=1) * 100).fillna(0) 
    # plt.figure(figsize=(12, 8))
    # plt.imshow(AR_sector_year_percentage, cmap="Reds", aspect="auto")
    # plt.colorbar(label="% of AR Reports")
    # plt.xticks(range(len(AR_sector_year_percentage.columns)), AR_sector_year_percentage.columns, rotation=45)
    # plt.yticks(range(len(AR_sector_year_percentage.index)), AR_sector_year_percentage.index)
    # plt.title("% of AR Reports per Sector per Year")
    # plt.xlabel("Year")
    # plt.ylabel("Sector")
    # plt.tight_layout()
    # plt.savefig("figures_part2/AR_sector_year_percentage_heatmap_v3.png")
    # plt.close()

    print("\nSECOTOR-SPECIFIC ANALYSIS COMPLETE. FIGURES SAVED IN figures_part2/ DIRECTORY")
  

if __name__ == "__main__":
    merged_df = load_merged_data()
    generate_analysis(merged_df)
    sector_specific_analysis(merged_df)
