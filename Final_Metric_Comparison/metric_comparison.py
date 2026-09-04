# TO UPLOAD


## this file is used to compare the file by Kerstin et al.
# titled esg_indicators_postprocessed.csv


# this file they created contains values for ESG indicators that they extracted
## refer to their paper to see the graphs they made; and their dashboard is extremely useful
## to seeing an overview per indicator

## highly recommend checking out Forster et al's (2026) dashboard:
# https://datastudio.google.com/reporting/ba729d7f-704d-4eb1-bb8d-ff3461cfdd2b/page/p_3h5kram1td?s=jUlW6L1X8u8
# -- can filter by indicator and see both the distribution for transparency + median value graph for each indicator


## essentially re-creating a portion of the graphs to to compare to my sentiment table
# selectively choosing ones that match my topics found in the bertopic bar chart of top 25 topics (reduced)

## for this; i selected the following topics from my bertopic output; mapped to the indicator i look at:
# Topic 1: emissions, energy, electriciy 
# Topic 2: water, waste, biodiversity  
# Topic 3: packaging, plastic, circular 
# Topic 5: climate, insurance, climate change 


import pandas as pd
import matplotlib.pyplot as plt


INPUT_FILE = "/"

SENTIMENT_TABLE_PATH = "/home/.../MRP/BERTopic_AN/sentiment_trend_percentages.parquet"

ESG_METRICS_PATH = "/home/.../MRP/Metric_Comparison/esg_indicators_postprocessed.csv"

OUTPUT_PATH = "/home/.../MRP/Metric_Comparison/output"


def get_metric_yearly_median(df, indicator_name):
    
    subset = df[df["data_point_description"] == indicator_name]

    yearly_median = subset.groupby("year")["value_final"].median()

    return yearly_median



def plot_trend_with_metric(ax, sentiment_table, topic_id, metric_series, topic_label, indicator_name):
    
    topic_sentiment = sentiment_table.loc[topic_id]

    ax.plot(topic_sentiment.index, topic_sentiment["opportunity"], color="green", marker="o", label="Opportunity Framing %")
    ax.set_ylabel("Opportunity Framing (%)", color="green")
    ax.set_xlabel("Year")
    ax.set_title(f"{topic_label} vs. {indicator_name}")

    ax_2 = ax.twinx()
    ax_2.plot(metric_series.index, metric_series.values, color="black", marker="o", linestyle="--", label="Median Metric Value")
    ax_2.set_ylabel("Median Metric Value", color="black")


if __name__ == "__main__":

    df = pd.read_csv(ESG_METRICS_PATH)
    # print("esg metrics df: \n")
    # print(df.head())
    # print(df.columns)
    # print(df.shape)
    # print(f"esg indicators df; fifth complete row: {df.iloc[5]}")


    ## focusing on the data point description column; which tells us about the indicator
    ## going to filter out some rows

    df = df[df["value_final"].notna()]

    ## going to get a list of the unique indicators
    # print(f"indicators used: {df['data_point_description'].unique().tolist()}")
    # description = df["data_point_description"].unique()

    ## searching based on the topics i selected
    # "scope 1", "scope 2", "scope 3", "emission", "emissions", "energy", "electricity", "scope", "co2",
    # "water", "waste", "biodiversity", "wastewater", "treatment",
    # "packaging", "plastic", "circular", "materials", "recycled",
    # for word in ["climate", "insurance", "climate change", "risk", "change"]:
    #     matches = [m for m in description if word.lower() in m.lower()]
    #     print(f"\nmatches for {word}\n")
    #     for m in matches:
    #         print(m)

    df["value_final"] = pd.to_numeric(df["value_final"], errors="coerce")


    sentiment_table = pd.read_parquet(SENTIMENT_TABLE_PATH)
    print(f"sentiment table df: \n{sentiment_table.head()}")
    print(sentiment_table.index.names)

    topics_to_compare = [
        {"topic_id": 1, "label": "Topic 1 (emissions/energy)", "indicator": "Gross Scope 1 greenhouse gas emissions"},
        {"topic_id": 2, "label": "Topic 2 (water/waste)", "indicator": "Water consumption"},
        {"topic_id": 3, "label": "Topic 3 (packaging/plastic)", "indicator": "Non-recycled waste"} ## changed from microplastics generated bc
        ## not evnough data points for microplastics generated
    ]

    fig, ax = plt.subplots(len(topics_to_compare), 1, figsize=(10, 5 * len(topics_to_compare)))

    for i in range(len(topics_to_compare)):
        topic = topics_to_compare[i]
        current_ax = ax if len(topics_to_compare) == 1 else ax[i]

        metric_series = get_metric_yearly_median(df, topic["indicator"])
        plot_trend_with_metric(current_ax, sentiment_table, topic["topic_id"], metric_series, topic["label"], topic["indicator"])

    fig.suptitle("Opportunity Trend vs. Reported Metric (by Topic)")
    plt.tight_layout()
    plt.savefig("/home/.../MRP/Metric_Comparison/sentiment_trend_vs_metric.png")
    plt.close()

    print("PLOT SAVED: sentiment_trend_vs_metric.png")



    ## topic 1: emissions, energy, electricity 
    # --> "Gross Scope 1 greenhouse gas emissions"
    # --> "Emissions to air"
    # --> "Energy consumption related to own operations"
    

    ## topic 2: water, waste, biodiversity
    # --> "Water consumption"
    # --> "Waste generated"
    # --> "Non-recycled waste"
    # --> "Hazardous waste"


    ## topic 3: packaging, plastic, circular
    # --> "Microplastics generated"
    # --> "Microplastics generated or used"

    
