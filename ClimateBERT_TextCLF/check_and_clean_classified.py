# TO UPLOAD

## checking the classified paragraphs

# manaully reading some of the paragraphs classified as "yes"
## seeing the  probabilities for these paragraphs

## might dro p some of the paragraphs that are classified as "yes" but have low clf probabilities
# (low confidence in the clf)


import pandas as pd

def drop_pars_interval(classified_df, bound):

    cleaned_classified_df = classified_df[classified_df['posterior_prob_score'] > bound]
    
    return cleaned_classified_df



if __name__ == "__main__":
    
    classified_df = pd.read_parquet("/home/.../MRP/ClimateBERT_TextCLF/classified_paragraphs.parquet")

    # print(f"total num of paragraphs classified as climate-related: {len(classified_df)}")
    # print(f"\ncolumns in the classified df: {classified_df.columns}\n")

    # print(f"\n\nSOME OF THE PARAGRAPHS CLASSIFIED AS CLIMATE-RELATED:\n\n{classified_df['text'].head(3).to_list()}\n\n")

    print(f"\nclf_label value counts:\n\n{classified_df['clf_label'].value_counts()}\n\n")

    ## checking for any duplicates
    # print(f"\n\nnum of duplicates in the classified df: {classified_df.duplicated().sum()}\n\n")

    ## checking for duplicate indexes
    # print(f"\n\nnum of duplicate indexes in the classified df: {classified_df.index.duplicated().sum()}\n\n")

    # print(f"\n\nlables and posterior probabilities for the first 3 paragraphs classified as climate-related:\n\n{classified_df[['clf_label', 'posterior_prob_score']].head(3)}\n\n")

    # print(f"\n\nnum of rows with score < 0.6: {len(classified_df[classified_df['posterior_prob_score'] < 0.6])}\n\n")

    # print(f"\n\nnum of rows with score < 0.7: {len(classified_df[classified_df['posterior_prob_score'] < 0.7])}\n\n")

    # print(f"\n\nnum of rows with score < 0.8: {len(classified_df[classified_df['posterior_prob_score'] < 0.8])}\n\n")

    # print(f"\n\nnum of rows with score < 0.9: {len(classified_df[classified_df['posterior_prob_score'] < 0.9])}\n\n")


    ## paragraphs with score >0.75 <0.8; checking if they are actually climate-related or not
    # print(f"\n\nnum of rows with score > 0.75 and < 0.8: {len(classified_df[(classified_df['posterior_prob_score'] > 0.75) & (classified_df['posterior_prob_score'] < 0.8)])}\n\n")

    # ## sampling text of these rows  
    # print(f"\n\ntext of the rows with score > 0.75 and < 0.8:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.75) & (classified_df['posterior_prob_score'] < 0.8)]['text'].sample(3).to_list()}\n\n")

    # print(f"\n\ntext of the rows with score > 0.8 and < 0.9:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.8) & (classified_df['posterior_prob_score'] < 0.9)]['text'].sample(3).to_list()}\n\n")

    # print(f"\n\ntext of the rows with score > 0.9 and < 0.95:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.9) & (classified_df['posterior_prob_score'] < 0.95)]['text'].sample(3).to_list()}\n\n")

    # print(f"\n\nnum of rows with score < 0.90: {len(classified_df[classified_df['posterior_prob_score'] < 0.90])}\n\n")

    # print(f"\n\ntext of rows with score > 0.85 and < 0.90:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.85) & (classified_df['posterior_prob_score'] < 0.90)]['text'].sample(3).to_list()}\n\n")


    # print(f"\n\nsample of rows with score > 0.8 and < 0.85:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.8) & (classified_df['posterior_prob_score'] < 0.85)]["text"].sample(3).to_list()}\n\n")

    # print(f"\n\nsample of rows with score > 0.85 and < 0.9:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.85) & (classified_df['posterior_prob_score'] < 0.9)]["text"].sample(3).to_list()}\n\n")


    # print(f"\n\nsample of rows with score > 0.9 and < 0.95:\n\n{classified_df[(classified_df['posterior_prob_score'] > 0.9) & (classified_df['posterior_prob_score'] < 0.95)]["text"].sample(3).to_list()}\n\n")

    # print(f"\n\nsample of rows with score > 0.95:\n\n{classified_df[classified_df['posterior_prob_score'] > 0.95]["text"].sample(3).to_list()}\n\n")

    # print(f"\n\nnum of rows (paragraphs) with score < 0.85: {len(classified_df[classified_df['posterior_prob_score'] < 0.85])}\n\n")

    # print(f"\n\nnum of rows (paragraphs) with score > 0.90: {len(classified_df[classified_df['posterior_prob_score'] > 0.90])}\n\n")
    # print(f"\n\nnum of rows (paragraphs) with score < 0.90: {len(classified_df[classified_df['posterior_prob_score'] < 0.90])}\n\n")


    # print(f"\n\nnum of rows with score < 0.95: {len(classified_df[classified_df['posterior_prob_score'] < 0.95])}\n\n")
    
    # print(f"\n\nnum of rows with score < 0.95 grouped by year:\n\n{classified_df[classified_df['posterior_prob_score'] < 0.95].groupby('year').size()}\n\n")


    ## dropping paragraphs with a score < 0.90
    cleaned_classified_df = drop_pars_interval(classified_df, 0.90)

    print(f"\n\ntotal num of paragraphs classified as climate-related after dropping paragraphs with score < 0.90: {len(cleaned_classified_df)}\n\n")

    ## svaing --> input for sentiment analysis
    cleaned_classified_df.to_parquet("/home/.../MRP/ClimateBERT_TextCLF/cleaned_classified_paragraphs.parquet", index=False)
    print(f"\n\ncleaned classified paragraphs SAVED!")
