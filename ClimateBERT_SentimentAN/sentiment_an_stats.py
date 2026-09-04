# TO UPLOAD

# STATISTICS to check snetiment clf results
# based on the firms (unique) etc. between the groups classified as risk vs opporutniy

 ## going to print stats for full df after joining with checkpoint file

# also going to visualize distribution of scores for each group (opportunity, risk, neutral) using graphs



import pandas as pd
import matplotlib.pyplot as plt



CHECKPOINT_FILE = "/home/.../MRP/ClimateBERT_SentimentAN/sentiment_clf_checkpoint_final2.csv"

INPUT_FILE = "/home/.../MRP/ClimateBERT_TextCLF/cleaned_classified_paragraphs.parquet"

def print_sentiment_stats_opportunity(full_df):

    opportunity_df = full_df[full_df['sentiment_label'] == 'opportunity']
    print(f"\n\nopportunity df shape: {opportunity_df.shape}\n\n")

    opportunity_firms = opportunity_df['firm'].nunique()
    print(f"\n\nnumber of unique firms classified as opportunity: {opportunity_firms}\n\n")

    opportunity_sectors = opportunity_df['primary_sics_sector'].nunique()
    print(f"\n\nnumber of unique sectors classified as opportunity: {opportunity_sectors}\n\n")

    opportunity_years = opportunity_df['year'].nunique()
    print(f"\n\nnumber of unique years classified as opportunity: {opportunity_years}\n\n")


    ## graph of number of paragraphs per primary_sics_sector for opportunity paragraphs
    plt.figure(figsize=(10, 6))
    opportunity_df.groupby('primary_sics_sector').size().sort_values(ascending=False).plot(kind='bar')
    plt.title("Number of Opportunity Paragraphs per Sector")
    plt.xlabel("Sector")
    plt.ylabel("Number of Opportunity Paragraphs")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig("/home/.../MRP/ClimateBERT_SentimentAN/graphs/opportunity_paragraphs_per_sector.png")
    plt.close()

    print(f"\n\nopportunity paragraphs by primary_sics_sector:\n{opportunity_df.groupby('primary_sics_sector').size().sort_values(ascending=False)}\n\n")

    ## saving opportunity df to parquet file in BERTOPI C folder
    opportunity_df.to_parquet("/home/.../MRP/BERTopic_AN/opportunity_df.parquet", index=False)
    print(f"\n\nopportunity df saved to parquet file in BERTOPIC folder\n\n")


def print_sentiment_stats_risk(full_df):
    risk_df = full_df[full_df['sentiment_label'] == 'risk']
    print(f"\n\nrisk df shape: {risk_df.shape}\n\n")

    risk_firms = risk_df['firm'].nunique()
    print(f"\n\nnumber of unique firms classified as risk: {risk_firms}\n\n")

    risk_sectors = risk_df['primary_sics_sector'].nunique()
    print(f"\n\nnumber of unique sectors classified as risk: {risk_sectors}\n\n")

    risk_years = risk_df['year'].nunique()
    print(f"\n\nnumber of unique years classified as risk: {risk_years}\n\n")

    print(f"\n\nrisk paragraphs by primary_sics_sector:\n{risk_df.groupby('primary_sics_sector').size().sort_values(ascending=False)}\n\n")

    ## graph of number of paragraphs per primary_sics_sector for risk paragraphs
    plt.figure(figsize=(10, 6))
    risk_df.groupby('primary_sics_sector').size().sort_values(ascending=False).plot(kind='bar')
    plt.title("Number of Risk Paragraphs per Sector")
    plt.xlabel("Sector")
    plt.ylabel("Number of Risk Paragraphs")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("/home/.../MRP/ClimateBERT_SentimentAN/graphs/risk_paragraphs_per_sector.png")
    plt.close()

    risk_df.to_parquet("/home/.../MRP/BERTopic_AN/risk_df.parquet", index=False)
    print(f"\n\nrisk df saved to parquet file in BERTOPIC folder\n\n")



def print_sentiment_stats_neutral(full_df):
    
    neutral_df = full_df[full_df['sentiment_label'] == 'neutral']
    print(f"\n\nneutral df shape: {neutral_df.shape}\n\n")

    neutral_firms = neutral_df['firm'].nunique()
    print(f"\n\nnumber of unique firms classified as neutral: {neutral_firms}\n\n")

    neutral_sectors = neutral_df['primary_sics_sector'].nunique()
    print(f"\n\nnumber of unique sectors classified as neutral: {neutral_sectors}\n\n")

    neutral_years = neutral_df['year'].nunique()
    print(f"\n\nnumber of unique years classified as neutral: {neutral_years}\n\n")

    print(f"\n\nneutral paragraphs by primary_sics_sector:\n{neutral_df.groupby('primary_sics_sector').size().sort_values(ascending=False)}\n\n")

    ## graph of number of paragraphs per primary_sics_sector for neutral paragraphs
    plt.figure(figsize=(10, 6))
    neutral_df.groupby('primary_sics_sector').size().sort_values(ascending=False).plot(kind='bar')
    plt.title("Number of Neutral Paragraphs per Sector")
    plt.xlabel("Sector")
    plt.ylabel("Number of Neutral Paragraphs")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("/home/.../MRP/ClimateBERT_SentimentAN/graphs/neutral_paragraphs_per_sector.png")
    plt.close()

    neutral_df.to_parquet("/home/.../MRP/BERTopic_AN/neutral_df.parquet", index=False)
    print(f"\n\nneutral df saved to parquet file in BERTOPIC folder\n\n")




if __name__ == "__main__":

    checkpoint_df = pd.read_csv(CHECKPOINT_FILE)

    print(f"\n\ncheckpoint file shape: {checkpoint_df.shape}\n\n")
    print(f"\n\ncheckpoint file head:\n{checkpoint_df.head(2)}\n\n")

    checkpoint_df = checkpoint_df.rename(columns={"predicted_label": "sentiment_label"})

    original_df = pd.read_parquet(INPUT_FILE)
    print(f"\n\noriginal df shape: {original_df.shape}\n\n")
    print(f"\n\noriginal df head:\n{original_df.head(2)}\n\n")

    og_index = set(range(len(original_df)))
    checkpoint_index = set(checkpoint_df['row_index'])
    assert og_index == checkpoint_index, f"\nOG index does NOT match checkpoint index; missing : {len(og_index - checkpoint_index)}, duplicates : {checkpoint_df['row_index'].duplicated().sum()}\n\n"

    checkpoint_df = checkpoint_df.sort_values("row_index").reset_index(drop=True)

    full_df = original_df.reset_index(drop=True)
    full_df["sentiment_label"] = checkpoint_df["sentiment_label"].values
    full_df["opportunity_score"] = checkpoint_df["opportunity_score"].values
    full_df["neutral_score"] = checkpoint_df["neutral_score"].values
    full_df["risk_score"] = checkpoint_df["risk_score"].values

    # ## joining the original df with the checkpoint file before neutral clf
    # full_df = original_df.merge(checkpoint_df, left_index=True, right_index=True, suffixes=("", "_chk"))

    # stats fo r full df
    print(f"\n\ncolumns of full df after joining with checkpoint file: {full_df.columns}\n\n")
    print(f"\n\nfull df shape: {full_df.shape}\n\n")
    print(f"\n\nfull df sentiment label counts:\n{full_df['sentiment_label'].value_counts()}\n\n")
    print(f"\n\nfull df head : {full_df.head(2)}\n\n")


    ## number of paragraphs per year for each sentiment label
    print(f"\n\nnumber of paragraphs per year for each sentiment label:")
    print(full_df.groupby(['year', 'sentiment_label']).size().unstack(fill_value=0))

    ## graph of number of paragraphs per year for each senitment lable
    # x-axis: year, y-axis: num of paragraphs, three bars for each year (opportunity, risk, neutral)
    plt.figure(figsize=(10, 6))
    label_order = ['opportunity', 'neutral', 'risk']
    colours = {'opportunity': 'tab:blue', 'neutral': 'tab:orange', 'risk': 'tab:red'}
    graph = full_df.groupby(['year', 'sentiment_label']).size().unstack(fill_value=0).reindex(columns=label_order)
    graph.plot(kind='bar', stacked=False, color=[colours[label] for label in label_order])
    plt.title("Number of Paragraphs per Year for Each Sentiment Label")
    plt.xlabel("Year")
    plt.ylabel("Number of Paragraphs")
    plt.legend(title="sentiment Label")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("/home/jasmine/masters/deep_learning/MRP/ClimateBERT_SentimentAN/graphs/paragraphs_per_year_sentiment_label.png")
    plt.close()

    ## overall distribution of sentiment labels (across ALL years)
    # creating a pie chart to visualize the distribution of sentiment labels
    plt.figure(figsize=(8, 8))
    label_order = ['opportunity', 'neutral', 'risk']
    colours = {'opportunity': 'tab:blue', 'neutral': 'tab:orange', 'risk': 'tab:red'}
    graph_pie = full_df['sentiment_label'].value_counts().reindex(label_order)
    graph_pie.plot(kind='pie', colors=[colours[label] for label in label_order], autopct="%1.1f%%")
    plt.title("Overall Distribution of Sentiment Labels")
    plt.ylabel("")  
    plt.legend(title="sentiment label", loc="best")
    plt.savefig("/home/.../MRP/ClimateBERT_SentimentAN/graphs/overall_distribution_sentiment_labels.png")
    plt.tight_layout()
    plt.close()



    # print_sentiment_stats_opportunity(full_df)
    # print_sentiment_stats_risk(full_df)
    # print_sentiment_stats_neutral(full_df)


    ## saving full df with sentiment labels to parquet file
    # full_df.to_parquet("/home/.../MRP/ClimateBERT_SentimentAN/full_df_sentiment_clf.parquet", index=False)

    ## saving opportunity, risk, neutral dfs to separate parquet files IN BERTOPIC folder
    # moved to do in function cals


## saves new directory with graphs for stats
# should also save new parquet file with full_df
## use ^ to splt into 3 groups (opportunity, risk, neutral) and save each group as a separate parquet file
