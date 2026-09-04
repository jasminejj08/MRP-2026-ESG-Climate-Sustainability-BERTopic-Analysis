# TO UPLOAD

## script to run bertopic on EACH group (opportunity, neutral, risk) individually
# --> analyzing each group indept to see waht each gorup talks about in particular

## UPDATED; producing the groups from the full_df_with_topics_reduced.parquet file
# rather than using the ones generated from sentiment clf step
## --> this is because i'm reusing the embeddings from the bertopic model run
# to enable comparison between the topics generated

## the 

EMBEDDINGS_FILE = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_embeddings.npy"

DF_FILE = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_with_reduced_topics.parquet"



import pandas as pd
import numpy as np
from bertopic import BERTopic
import time
from bertopic.representation import KeyBERTInspired

from hdbscan import HDBSCAN
from umap import UMAP
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import os
import torch

from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt






def run_bertopic_on_group(docs, embeddings, embedding_model, timestamps, min_cluster_size, group_name, output_dir, figures_dir):

    print("\nruning BerTopic on group: \t", group_name)
    start_time = time.time()

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_method="eom", prediction_data=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired()

    topic_model.update_topics(docs, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, representation_model=representation_model)
    print("\ntopic representation UPDATED\n")

    topic_model.reduce_topics(docs, nr_topics=25)
    print(f"\ntopics REDUCED to {topic_model.get_topic_info().shape[0]} topics\n")

    print("\n\nlen of docs and topic_model.topics_: ", len(docs), len(topic_model.topics_))


    print("\n\nfinished running BERTopic on group: \t", group_name)
    print("time taken: ", time.time() - start_time, " seconds")
    print(f"\n{group_name} num of topics: ", {topic_model.get_topic_info().shape[0]})


    model_save_path = os.path.join(output_dir, f"{group_name}_bertopic_MODEL")
    topic_model.save(model_save_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=True)

    print("\nNUMBER OF TOPICS (global) before reduction: \n")
    print(topic_model.get_topic_info().shape[0])


    ## generating figures for this group
    print(f"\n\nGenerating figures for group: \t{group_name}")


    print(f"\n len of docs: {len(docs)}; len of timestamps: {len(timestamps)}; len of topics: {len(topics)}\n")
    print("\nrunning topics over time\n")
    topics_over_time = topic_model.topics_over_time(docs, timestamps, topics=topic_model.topics_)
    print("\ntopics over time generated\n")

    ## figures
    fig1 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, title=f"Topics Over Time ({group_name}) Top 10 Topics")
    fig1.write_html(os.path.join(figures_dir, f"{group_name}_topics_over_time.html"))

    ## normalized topics over time
    fig4 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, normalize_frequency=True, title=f"Topics Over Time ({group_name}): Top 10 (normalized)")
    fig4.write_html(os.path.join(figures_dir, f"{group_name}_normalized_topics_over_time.html"))

    fig2 = topic_model.visualize_barchart(top_n_topics=10, title=f"Top 10 Topics ({group_name})")
    fig2.write_html(os.path.join(figures_dir, f"{group_name}_barchart.html"))

    fig3 = topic_model.visualize_heatmap(title=f"Topic Similarity Heatmap ({group_name})")
    fig3.write_html(os.path.join(figures_dir, f"{group_name}_heatmap.html"))


    print("\n\nDONE generating figures; returning to main function\n\n")


    return topic_model, topic_model.topics_
    




if __name__ == "__main__":

    full_df = pd.read_parquet(DF_FILE)

    loaded_embeddings = np.load(EMBEDDINGS_FILE)
    
    assert len(full_df) == len(loaded_embeddings), "number of ws in full_df and loaded_embeddings do NOT match"

    print("\n\nnumber of rows in full_df: ", len(full_df))
    
    opportunity_mask = (full_df['sentiment_label'] == 'opportunity').values
    neutral_mask = (full_df['sentiment_label'] == 'neutral').values
    risk_mask = (full_df['sentiment_label'] == 'risk').values

    opportunity_df = full_df[opportunity_mask].reset_index(drop=True)
    neutral_df = full_df[neutral_mask].reset_index(drop=True)
    risk_df = full_df[risk_mask].reset_index(drop=True)

    opportunity_embeddings = loaded_embeddings[opportunity_mask]
    neutral_embeddings = loaded_embeddings[neutral_mask]
    risk_embeddings = loaded_embeddings[risk_mask]


    print("\n\nnumber of rows in opportunity_df: ", len(opportunity_df))
    print("\nnumber of rows in neutral_df: ", len(neutral_df))
    print("\nnumber of rows in risk_df: ", len(risk_df))

    print("\n\nnumber of rows in opportunity_embeddings: ", len(opportunity_embeddings))
    print("\nnumber of rows in neutral_embeddings: ", len(neutral_embeddings))
    print("\nnumber of rows in risk_embeddings: ", len(risk_embeddings))

    print(opportunity_df.shape[0], neutral_df.shape[0], risk_df.shape[0])
    print(opportunity_df.shape[0] + neutral_df.shape[0] + risk_df.shape[0], "should equal", full_df.shape[0])

    ## each group might have diff min_cluster_size param in hdbscan to account for different number of rows in each group
    

    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')


    ## the min_clusteR_size param is set according to the amount of rows in each group;
    # and based off of the total number of rows used in the full bertopic run

    opp_model, opp_topics = run_bertopic_on_group(
        docs = opportunity_df['text'].tolist(),
        embeddings = opportunity_embeddings,
        embedding_model = sentence_model,
        timestamps = opportunity_df['year'].tolist(),
        min_cluster_size = 30, ## trying out half since corpora is around half
        group_name = "opportunity",
        output_dir = "/home/.../MRP/BERTopic_AN/group_Outputs",
        figures_dir = "/home/.../MRP/BERTopic_AN/group_Figures"
    )

    opportunity_df['topic'] = opp_topics
    opportunity_df.to_parquet("/home/.../MRP/BERTopic_AN/group_Outputs/opportunity_df_with_topics.parquet", index=False)


    neutral_model, neutral_topics = run_bertopic_on_group(
        docs = neutral_df['text'].tolist(),
        embeddings = neutral_embeddings,
        embedding_model = sentence_model,
        timestamps = neutral_df['year'].tolist(),
        min_cluster_size = 50, ## trying out half since corpora is around half
        group_name = "neutral",
        output_dir = "/home/.../MRP/BERTopic_AN/group_Outputs",
        figures_dir = "/home/.../MRP/BERTopic_AN/group_Figures"
    )
    neutral_df['topic'] = neutral_topics
    neutral_df.to_parquet("/home/jasmine/masters/deep_learning/MRP/BERTopic_AN/group_Outputs/neutral_df_with_topics.parquet", index=False)

    risk_model, risk_topics = run_bertopic_on_group(
        docs = risk_df['text'].tolist(),
        embeddings = risk_embeddings,
        embedding_model = sentence_model,
        timestamps = risk_df['year'].tolist(),
        min_cluster_size = 15, ## trying out half since corpora is around half
        group_name = "risk",
        output_dir = "/home/.../MRP/BERTopic_AN/group_Outputs",
        figures_dir = "/home/.../MRP/BERTopic_AN/group_Figures"
    )
    risk_df['topic'] = risk_topics
    risk_df.to_parquet("/home/.../MRP/BERTopic_AN/group_Outputs/risk_df_with_topics.parquet", index=False)


    print("\n\nDONE running BERTopic on EACH group (opportunity, neutral, risk) individually;\n\n")



