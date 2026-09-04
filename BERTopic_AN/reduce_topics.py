# TO  UPLOAD

# reducing topics after the full run (after trainigng the BERTopic) model

# documetnation:
# https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html#topic-reduction-after-training


## already ran bertopic_all_ver2.py to train model on entire corpus and saved the model and embeddings; 

# total num fo topics was aroudn 450; going to reduce this 

## also going to overwrite the visualizations and figs from full run with the reduced topics


import pandas as pd
import numpy as np
from bertopic import BERTopic
from umap import UMAP
from wordcloud import WordCloud
import matplotlib.pyplot as plt


MODEL_PATH = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/bertopic_all_corpus_MODEL"

DOCS_PATH = "/home/.../MRP/ClimateBERT_SentimentAN/full_df_sentiment_clf.parquet"

EMBEDDINGS_PATH = "/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_embeddings.npy"


def create_viz(topic_model, full_df, embeddings):
   
    docs = full_df["text"].tolist()
    timestamps = full_df["year"].tolist()

    print("\nRE-running topics over time\n")
    topics_over_time = topic_model.topics_over_time(docs, timestamps, topics=topic_model.topics_)
    print("\ntopics over time generated\n")

   
    print("\n\nvisualizing topics over time and other visualizations now\n")
    ##  visual of top 10 topics over time
    fig1 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, title="Topics Over Time (All Corpus): Top 10")
    fig1.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_all_topics_over_time_top_10_reduced.html")

    fig1_2 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, normalize_frequency=True, title="Topics Over Time (All Corpus): Top 10 (normalized)")
    fig1_2.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_all_topics_over_time_top_10_normalized_reduced.html")


    ## barchart of top 10 topics (overall NOT over time)
    fig2 = topic_model.visualize_barchart(top_n_topics=10, n_words=5, title="Top 10 Topics (All Corpus) (Reduced)")
    fig2.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_all_barchart_top_10_reduced.html")

    ## heatmap of topic's similarity matrix (similarity between topics) overall NOT over time
    fig3 = topic_model.visualize_heatmap(top_n_topics=10, title="Topic (Reduced) Similarity Heatmap (All Corpus): Top 10")
    fig3.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_all_heatmap_top_10_reduced.html")

    # fig4 = topic_model.visualize_term_rank(topics=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], title="Term Score Decline per Topic (All Corpus): Top 10")
    # fig4.write_html("/home/jasmine/masters/deep_learning/MRP/BERTopic_AN/all_corpus_Reduced_Figures/bertopic_all_term_rank_top_10.html")

    ## also doing topics per class where class = sentiment label (opportunity, neutral, risk)
    print("\n\nrunning topics per class now\n")
    topics_per_class = topic_model.topics_per_class(docs, classes=full_df["sentiment_label"].tolist())
    print("\ntopics per class generated\n")
    # print(f"\ntopics per class (head 5): \n {topics_per_class.head(5)}")

    fig5 = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10, title="Topics (Reduced) per Class (All Corpus): Top 10")
    fig5.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_all_topics_per_class_top_10_reduced.html")


    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    fig6 = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True, sample=0.1,title="Documents and Topics (All Corpus) (sampled 10%)")
    fig6.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures/bertopic_all_documents_viz_reduced.html")


    print("\ncreating wordcloud for most frequent topic (topic 0)\n")
    wc_topic = 0
    text = {word: value for word, value in topic_model.get_topic(wc_topic)}
    wc = WordCloud(width=800, height=400, background_color="white", max_words=1000).generate_from_frequencies(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("/home/.../MRP/BERTopic_AN/all_corpus_Reduced_Figures_2/bertopic_allcorpus_topic0_wordcloud.png")
    plt.close()


    




if __name__ == "__main__":
    
    loaded_topic_model = BERTopic.load(MODEL_PATH)

    full_df = pd.read_parquet(DOCS_PATH)

    loaded_embeddings = np.load(EMBEDDINGS_PATH)


    print("\n\nNUMBER OF TOPICS (global) before reduction: \n")
    print(loaded_topic_model.get_topic_info().shape[0])

    ## reduce topics
    ## selected 25 through iterative testing; 
    loaded_topic_model.reduce_topics(full_df["text"].tolist(), nr_topics=25)

    print("\n\nNUMBER OF TOPICS (global) after reduction: \n")
    print(loaded_topic_model.get_topic_info().shape[0])

    full_df["topic_reduced"] = loaded_topic_model.topics_
    full_df.to_parquet("/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_with_reduced_topics.parquet", index=False)

    # print("\n\ntopics:")
    # topics = loaded_topic_model.topics_
    # print(topics)

    create_viz(loaded_topic_model, full_df, loaded_embeddings)

    loaded_topic_model.save("/home/.../MRP/BERTopic_AN/all_corpus_Outputs/bertopic_all_corpus_MODEL_reduced", serialization="safetensors", save_ctfidf=True, save_embedding_model=True)

    print("\n\nDONE; reduced topics + visualizations created\n")


