# MRP-2026-ESG-BERTopic-Analysis
This repository is for my 2026 MRP at TMU, applying BERTopic to ESG disclosures through a designed pipeline to determine temporal themes and possible greenwashing.

---

### Abstract
ESG disclosures and reporting have become an indispensable part of decision-making, both for investors and corporations as we pay more attention to sustainability and environmental contributions. This paper aims to complement current research by taking a closer look at the linguistic patterns and themes pertaining to the E (environmental) pillar, specifically the climate domain, to analyze the language temporally. Such opportunity-framed climate-related language, which we coin as “green language”, is studied to track drift over time by analyzing how companies talk about what they report. By applying ClimateBERT and BERTopic to select sustainability reports from European firms over a 10 year period, we observe how topics, themes, and sentiments have changed over time and use these changes to highlight a viable method to measure potential greenwashing all by using internal, firm-disclosed data.

Key Words: ESG, environment, climate, sustainability, climateBERT, BERTopic, dynamic topic modeling, linguistic analysis 

--- 

### Dataset Credits 
All credits for the dataset used are attributed to the original authors of the paper titled "Assessing corporate sustainability with large language models: Evidence from Europe" (Forster et al., 2026). In particular, all credits rightfully belong to to Kerstin Forster, Lucas Keil, Victor Wagner, Maximilian A. Müller, Thorsten Sellhorn, Stefan Feuerriegel. No ownership of the data is claimed by us.  

---

### Environment and Machine Specifications
```
Python 3.12.3 
Ubuntu 24.04 LTS (WSL2) 
NVIDIA GeForce RTX 4060 GPU 
```
--- 

### Directory Structure

```
MRP (root directory)
│ 
├── EDA
│      └── eda_part1.py
│      └── eda_part2.py
│      └── /figures
│      └── /figures_part2
│      └── sampling.py
│      └── companies.csv
│      └── report_ids.csv
│      └── reports_per_company_year.csv
│      └── merged_companies_reports_per_company_year.csv
│
├── Sampled_Files
│      └── /Extracted_Paragraphs_Final
│               └── exploded_paragraphs.parquet
│               └── clean_extracted_text.py
│               └── check_extracted_paragraphs.py
│               └── english_only_df.parquet
│      └── all_SR_reports
│      └── final_df.csv
│
├── Extracted_Text
│      └── /ExtractPragraphsScripts
│               └── extract_pargraphs_PARQUETver.py
│               └── extract_paragraphs_parallel.py
│
├── ClimateBERT_TextCLF
│      └── climatebert_textclf.py
│      └── classified_paragraphs.parquet
│      └── check_and_clean_classified.py
│      └── checkpoint_file_classified.csv
│      └── cleaned_classified_paragraphs.parquet
│
├── ClimateBERT_SentimentAN
│      └── climatebert_sentiment_finalver.py
│      └── sentiment_clf_checkpoint_final.csv
│      └── sentiment_an_stats.py
│      └── full_df_sentiment_clf.parquet
│      └── /graphs
│
├── BERTopic_AN
│      └── bertopic_all_ver2.py
│      └── /all_corpus_Figures
│      └── /all_corpus_Outputs
│      └── /all_corpus_Reduced_Figures
│      └── bertopic_groups.py
│      └── /group_Figures
│      └── analyze_sentiment_trends.py
│
├── Final_Metric_comparison
│      └── metric_comparison.py
│      └── esg_indicators_postprocessed.csv 
```

---

### Conclusions
By applying
ClimateBERT and BERTopic to select sustainability reports from European firms over a 10 year period,
we observe how topics, themes, and sentiments have changed over time and use these changes to
highlight a viable method to measure potential greenwashing all by using internal, firm-disclosed data. We found that while the main topics of climate-related text within the sustainability reports of the data were dominated by keywords sustainability, esg, emissions, energy, electricity over time, topics represented by keywords food, farmers, oil, deforestation, climate, climate change and risk greatly increased after 2019. We also find that while sentiment framing and self-disclosed metrics align for some topics such as water consumption, others like packaging and waste show periods of divergence between opportunity-framed language and actual performance, suggesting potential greenwashing signals. These findings support a viable method to measure potential greenwashing using internal, firm-disclosed data alone. 

