# TO UPLOAD

# analyzing the full_df with topics from bertopic_all_ver2.py

##using the full_df with topics to analyze the sentiment trend for each topic over time
# how does the way companies talk about a topic change over time --> does the sentiment shift from one to another
# ex. opportunity to risk;

## topics over time itself tells us how much a topic is discussed over time;
## and topics per class tells us how much a topic is discussed in each class (opportunity, risk, neutral)

## this analysis kind of combines the two; we can see how the sentiment trend is for each topic over time
## --> let's us see if a topic is becoming more "opportunity" or more "risk" over time;

## sentiment table;

## going to foucs on the top 10 topics (total topics overall = 25 (from reduced))


##3 snetiment table; what fraction of the paragraphs in a topic are opportunity, risk, or neutral; 
## --> focusing on the sentiment (tone; linguistic style) of the paragraphs in a topic; not the amount of paragraphs in a topic
## -- topics_over_time might show a topic stay level over time, but the sentiment of that topic might change over time; ex. a topic might be mostly opportunity in 2010, but mostly risk in 2020; this is what we want to see

## for pot greenwashing, if a topic's language becomes more "opportunity" over time, but metrics show otherwise;


## working with REDUCED full_df with topics (2t topics)


import pandas as pd
import matplotlib.pyplot as plt
import math
from bertopic import BERTopic


INPUT_FILE = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_with_reduced_topics.parquet"

MODEL_PATH = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/bertopic_all_corpus_MODEL_reduced"


def analyze_df(full_df_topics, topic_model):
    
    ## want to see how the sentiment trend is for each topic over time; can use this to see if a topic is becoming more positive or negative over time
    ## going to use the full_df with topics
    # can use this to compare to metrics by Kirstin et al. after
    ## group by topic, year, sentiment_label; then unstack sentiment_label to get counts of each sentiment per topic per year 
    
    counts_per_label = full_df_topics.groupby(["topic_reduced", "year", "sentiment_label"]).size()
    counts_per_label = counts_per_label.unstack("sentiment_label", fill_value=0)

    print(f"\ncounts_per_label: \n{counts_per_label.head()}")

    ## cecking topic 23 bc there was an issue with the graph;
    print(f"\n counts per label for topic 23: \n{counts_per_label.loc[23]}")

    ## normalize across sentiment labe
    percentages = counts_per_label.div(counts_per_label.sum(axis=1), axis=0)
    percentages = percentages[percentages.index.get_level_values("topic_reduced") != -1]
    print(f"\npercentages: \n{percentages.head()}")

    ## saving the percentages to parquet file
    percentages.to_parquet("/home/.../MRP/BERTopic_AN/sentiment_trend_percentages.parquet")


    # ## plotting a graph of the sentiment trend
    # ## looking at top 5 topics 
    # # topics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # ## looking at select topics i saw from the barchart;
    # topics_selected = []

    # ## going to do ALL 25 topics from reduced
    # topics = sorted(percentages.index.get_level_values("topic_reduced").unique())

    # ## plotting a graph for each topic in topics;
    # ## will be a line graph with x-axis = year, y0-axis = percentage of paragraphs for each sentiment label
    # ## 3 lines on each graph; one for each sentiment label (opportunity, risk, neutral)

    # num_topics = len(topics)
    # num_cols = 5
    # num_rows = math.ceil(num_topics / num_cols)
    # # num_rows = 5

    # print(f"\n\nnum_topics: {num_topics}, num_cols: {num_cols}, num_rows: {num_rows}\n\n")

    # fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 4 * num_rows), sharey=True)
    # ax = ax.flatten()

    # for i in range(len(topics)):
    #     current_topic = topics[i]
    #     current_ax = ax[i]

    #     topic_percentages = percentages.loc[current_topic]

    #     years = topic_percentages.index
    #     opportunity_percentages = topic_percentages["opportunity"]
    #     neutral_percentages = topic_percentages["neutral"]
    #     risk_percentages = topic_percentages["risk"]

    #     current_ax.plot(years, opportunity_percentages, marker='o', label="opportunity", color="green")
    #     current_ax.plot(years, neutral_percentages, marker='o', label="neutral", color="orange")
    #     current_ax.plot(years, risk_percentages, marker='o', label="risk", color="red")


    #     topic_label = get_topic_label(topic_model, current_topic, 3)
    #     current_ax.set_title(topic_label)

    #     # current_ax.set_title(f"Topic: {current_topic}")
    #     current_ax.set_xlabel("Year")
    #     # current_ax.set_ylabel("Percentage of Paragraphs")
    #     # current_ax.legend()
    #     current_ax.grid(True)
    
    # for j in range(len(topics), len(ax)):
    #     ax[j].axis('off')
    
    # ax[0].set_ylabel("Percent of Paragraphs")
    # ax[0].legend()
    # plt.suptitle("Sentiment Trend for Top 25 Topics", fontsize=16)
    # plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.savefig("/home/.../MRP/BERTopic_AN/sentiment_trend_25_topics.png", dpi=150)
    # plt.close()


def get_topic_label(topic_model, topic_id, n_words=3):
    ## for better visualization on the graphs; going to attach the top 3
    # words for each topic to the title of the graph
    topic_words = topic_model.get_topic(topic_id)
    words = [word for word, score in topic_words[:n_words]]
    return f"Topic {topic_id}: " + ", ".join(words)


if __name__ == "__main__":
    
    full_df_topics = pd.read_parquet(INPUT_FILE)

    print("\n\nnumber of topics: ", full_df_topics["topic_reduced"].nunique()) ## shouldb e 25 (from reduced)

    topic_model = BERTopic.load(MODEL_PATH)

    analyze_df(full_df_topics, topic_model)



