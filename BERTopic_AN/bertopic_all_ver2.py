# TO UPLOAD

## script for running BERTopic on ALL groups consecutively

## using dynamic topic modeling; refer to documentation 

## UPDATED: not using gpu acceleraiton for umap + hdbscan since some library/environment issue



## updated; this file will apply bertopic to the entire corpus/output of sentiment clf
## (all paragraphs as one big parquet file; saved in the prev step)
# then anothe file will apply bertopic to each group separately
## compare this to see

# relevant documention:
# https://maartengr.github.io/BERTopic/getting_started/topicsovertime/topicsovertime.html
# https://huggingface.co/docs/hub/en/bertopic
# https://maartengr.github.io/BERTopic/getting_started/embeddings/embeddings.html
# https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.get_topic_freq
# https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.get_topic_freq
# https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html?utm_source=chatgpt.com#removing-stop-words


# also refering to the original model documentations for further arguments/parametres
# https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
# https://umap-learn.readthedocs.io/en/latest/parameters.html


## each group will have a bertopic model trained+fited to it ## --> MOVED TO SEPARATE FILE


import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
# from cuml.cluster import HDBSCAN ## for gpu acceleration
# from cuml.manifold import UMAP ## for gpu acceleration
## gpu accelerator not wroking; libraries not compatible with current stettup

from hdbscan import HDBSCAN
from umap import UMAP

from wordcloud import WordCloud
import time
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import torch
from bertopic.vectorizers import ClassTfidfTransformer


INPUT_FILE = "/home/.../MRP/ClimateBERT_SentimentAN/full_df_sentiment_clf.parquet"



def run_bertopic(full_df):

    ## rfer to bertopic, hdbscan, umap documentations for parameter

    ## updated to use update_topics() instead of passing vectorizer_model and ctfidf_model to BERTopic() init

    ## docs = paragraphs
    docs = full_df["text"].tolist()
    timestamps = full_df["year"].tolist()

    assert len(docs) == len(timestamps), "timsetamps do NOT match len of docs;"


    ## pass in embeddings, umap, hdbscan to BERTopic, fit on docs
    ## customizable/parameter tuning can be done here 
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda") ## embedings will be done on gpu
    embeddings = sentence_model.encode(docs, show_progress_bar=True, batch_size=128, device="cuda")

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42) ## add random_state if reproducibility
    
    ## i increased the min_cluster_size (default 10); tuning this helped reduce num of topics 
    hdbscan_model = HDBSCAN(min_cluster_size=100, min_samples=5, cluster_selection_method='eom', metric='euclidean', prediction_data=True)

    ## hdbscan created the clusters; going to fit bertopic model on the docs and embeddings; then will update the topic rep with vectorizer + ctfidf
    ## updated to not pass vectorizer_model, ctfidf_model, representation_model --> finetuned after
    topic_model = BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        # nr_topics="auto",
        verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    print("\nmodel fitted; topics + probs generated;\n")

    print("\n fine-tuning topic representation now\n")
    ## topic re-presentation update
    # fine-tuning topic reps by AFTER training model
    ## vectorizer model; allow 2-grams for topic rep; disable stop words; 
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired()

    topic_model.update_topics(docs, vectorizer_model=vectorizer_model, ctfidf_model=ctfidf_model, representation_model=representation_model)
    print("\ntopic representation UPDATED\n")

    ## topics over time (dynamitc topic modeling)
    # need thes to have same length 
    print(f"\n len of docs: {len(docs)}; len of timestamps: {len(timestamps)}; len of topics: {len(topics)}\n")
    print("\nrunning topics over time\n")
    topics_over_time = topic_model.topics_over_time(docs, timestamps, topics=topics)
    print("\ntopics over time generated\n")

    print("\n\nsaving embeddings for re-use + comparison\n")
    np.save("/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_embeddings.npy", embeddings)
    ## append topics + probs to the df and save as another df
    full_df_2 = full_df.copy()
    full_df_2["topic"] = topics
    full_df_2["topic_prob"] = probs
    full_df_2.to_parquet("/home/.../MRP/BERTopic_AN/all_corpus_Outputs/full_df_with_topics.parquet", index=False)
    print("\nfull_df with topics + probs saved to parquet\n")
    topic_model.save("/home/.../MRP/BERTopic_AN/all_corpus_Outputs/bertopic_all_corpus_MODEL", serialization="safetensors", save_ctfidf=True, save_embedding_model=True)
    print("\nmodel saved\n")

    ### now can do topic representation update, print statistics, and run topics over time

    print("\n\nprinting statistics + visualizing topics now\n")
    
    print("\nNUMBER OF TOPICS (global) before reduction: \n")
    print(topic_model.get_topic_info().shape[0])

    # print("\nALL TOPICS: \n")
    # print(topic_model.get_topics())
    
    print("TOPIC INFO (head 10): \n")
    print(topic_model.get_topic_info().head(10))

    print("\nTOPIC FREQUENCY (head 10): \n")
    print(topic_model.get_topic_freq().head(10))

    # print("\nREPRESENTATIVE DOCS (heaad 10): \n")
    # ## updated
    # reps = topic_model.get_representative_docs()
    # for i, (topic, rep_docs) in enumerate(reps.items()):
    #     if i >= 10:
    #         break
    #     print(f"topic {topic}: {rep_docs}\n")

    ### only topics_over_time is a temporal analysis; the rest is called on the global topics (not over time)

    print("\n\nvisualizing topics over time and other visualizations now\n")
    ##  visual of top 10 topics over time
    fig1 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, title="Topics Over Time (All Corpus): Top 10")
    fig1.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_topics_over_time_top_10.html")

    fig1_2 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10, normalize_frequency=True, title="Topics Over Time (All Corpus): Top 10 (normalized)")
    fig1_2.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_topics_over_time_top_10_normalized.html")


    ## barchart of top 10 topics (overall NOT over time)
    fig2 = topic_model.visualize_barchart(top_n_topics=10, n_words=5, title="Top 10 Topics (All Corpus)")
    fig2.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_barchart_top_10.html")

    ## heatmap of topic's similarity matrix (similarity between topics) overall NOT over time
    fig3 = topic_model.visualize_heatmap(top_n_topics=10, title="Topic Similarity Heatmap (All Corpus): Top 10")
    fig3.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_heatmap_top_10.html")

    fig4 = topic_model.visualize_term_rank(topics=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], title="Term Score Decline per Topic (All Corpus): Top 10")
    fig4.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_term_rank_top_10.html")

    ## also doing topics per class where class = sentiment label (opportunity, neutral, risk)
    print("\n\nrunning topics per class now\n")
    topics_per_class = topic_model.topics_per_class(docs, classes=full_df["sentiment_label"].tolist())
    print("\ntopics per class generated\n")
    # print(f"\ntopics per class (head 5): \n {topics_per_class.head(5)}")

    fig5 = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10, title="Topics per Class (All Corpus): Top 10")
    fig5.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_topics_per_class_top_10.html")


    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
    fig6 = topic_model.visualize_documents(docs, reduced_embeddings=reduced_embeddings, hide_document_hover=True, hide_annotations=True, sample=0.1,title="Documents and Topics (All Corpus) (sampled 10%)")
    fig6.write_html("/home/.../MRP/BERTopic_AN/all_corpus_Figures/bertopic_all_documents_viz.html")


    print("\ncreating wordcloud for most frequent topic (topic 0)\n")
    wc_topic = 0
    text = {word: value for word, value in topic_model.get_topic(wc_topic)}
    wc = WordCloud(width=800, height=400, background_color="white", max_words=1000).generate_from_frequencies(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.savefig("/home/jasmine/masters/deep_learning/MRP/BERTopic_AN/all_corpus_Figures/bertopic_allcorpus_topic0_wordcloud.png")
    plt.close()

    print("\n\nDONE VISUALIZATIONS + STATISTICS; returning to main func \n")



def prepare_data(full_df):
    ## sampling and checking that any additional cleaning needs to be done):
    print(f"sampling of df before bertopic: \n {full_df['text'].sample(20).to_list()}\n")
    


def df_stats_check(full_df):
    print("checking stats of loaded df\n")
    print(f"columns: {full_df.columns}\n")
    print(f"shape: {full_df.shape}\n")
    print("")




if __name__ == "__main__":

    ## load data, load model, apply model to each group separately but load model only once
    ## using methods for modularity again
    # opportunity_df = pd.read_parquet(OPPOR_INPUT_FILE)
    # neutral_df = pd.read_parquet(NEU_INPUT_FILE)
    # risk_df = pd.read_parquet(RISK_INPUT_FILE)

    full_df = pd.read_parquet(INPUT_FILE)

    # test_df = full_df.sample(100000, random_state=42).reset_index(drop=True)

    df_stats_check(full_df)

    prepare_data(full_df)

    # print(torch.cuda.is_available(), torch.randn(4,4).cuda())

    # HDBSCAN(min_cluster_size=5).fit(np.random.rand(100, 10).astype("float32"))

    os.makedirs("/home/.../MRP/BERTopic_AN/all_corpus_Outputs", exist_ok=True)
    os.makedirs("/home/.../MRP/BERTopic_AN/all_corpus_Figures", exist_ok=True)

    print("\nREADY\n")

    # ## TESTING
    # test_time_start = time.time()
    # run_bertopic(test_df)
    # print(f"bertopic TEST completed in {(time.time() - test_time_start) / 60} minutes")

    print("running bertopic on entire corpus now\n")
    time_start = time.time()

    run_bertopic(full_df)
    print(f"bertopic completed in {(time.time() - time_start) / 60} minutes")

    print("done with bertopic on entire corpus;\n")


## my notes on sep doc
