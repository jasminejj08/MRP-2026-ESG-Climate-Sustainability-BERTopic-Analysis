## 
## TO UPLOAD

# extracting paragraphs from the pdf report files
## curent path to reports: /home/jasmine/masters/deep_learning/MRP/Extracted_Files/

## want to get USABLE textual data from the pdf files for the ClimateBERT

## things to note:
## don't want to extract tables, figures, captions, headers, footers etc.
## don't want to extract too short paragraphs
## also no numbers (focusing on textual data --> LINGUISTICS)

## process: extracting text from pdf files, then chucnking into paragraphs
# final stored in csv file --> each row = 1 paragraph
# going to take care of tables, figures etc. DURING extraction process 

## library chosen: pymupdf --> most fast 

import os
import re
import unicodedata
import pymupdf # pip install pymupdf
import pandas as pd
import time

## setting the max and min based on ClimateBERT's max token length ==> 512
MIN_PARAGRAPH_LENGTH = 50 ## minimum number of words in a paragraph 
MAX_PARAGRAPH_LENGTH = 350



def check_exploded_df_csv(exploded_df_file):
    ## checking before processing
    if not os.path.exists(exploded_df_file):
        raise FileNotFoundError(f"Exploded DataFrame CSV file not found: {exploded_df_file}")


    exploded_df = pd.read_csv(exploded_df_file)

    print(f"exploded df csv shape: {exploded_df.shape}")
    print(f"exploded df csv head:\n {exploded_df.head()}")

    return exploded_df



def get_usable_text(page):

    ## using pymupdf's built-in functions 
    # get bounding boxes of tables and then filter out text blocks that intersect with those bounding boxes

    ## tables == all tables in current page
    tables = page.find_tables(strategy="lines_strict")

    ## table_bboxes == boudning boxes of all tables in current page
    table_bboxes = [pymupdf.Rect(t.bbox) for t in tables.tables]

    ## get all text blocks in current page (coordinates of each block, text, block number, block type)
    blocks = page.get_text("blocks", sort=True)

    keep_text = [] ## will only kep text that does not intesesect with any table bounding box --> usable text


    for block in blocks:

        x0, y0, x1, y1, block_text, block_no, block_type = block ## unpacking

        if block_type != 0: ## 0 is text block, 1 is image, 2 is vector graphics, 3 is line, 4 is rectangle
            continue

        ## the bounding box of current text block
        block_rect = pymupdf.Rect(x0, y0, x1, y1)

        if any(block_rect.intersects(table_bbox) for table_bbox in table_bboxes):
            continue ## skipping this block if intersects with a table bounding box

        keep_text.append(block_text)

    return "\n\n".join(keep_text)



def classify_page(page):

    ## pymupdf has functions; get_table() and get_images() to extract tables and images from the pdf file
    ## --> using these to filter out pages tha are table-heavy or image-heavy 

    text = get_usable_text(page)

    ## need to get rid of text too short
    ## also table of content pages often have "..." or such so gonna get rid of that


    if len(text.strip()) < 100:
        return {"text": text, "label": "NOTUSABLE"}

    dots = len(re.findall(r"\.{3,}", text))
    if dots > 10:
        return {"text": text, "label": "NOTUSABLE"}
    
   
    return {"text": text, "label": "usable"}



# # eopning the pdf file, going page by page to first configure which pages can actually be used
## only those extracted pages will then be chunked into paragraphs 

def extract_pages(pdf_path):
    ## extract USABLE pages from the current pdf file --> pages that are not table-heavy or junk
    
    ## list to store usable pages
    pages = []

    current_document = pymupdf.open(pdf_path)
    # print(f"CURRENT DOCUMENT: {pdf_path}, total pages: {len(current_document)}")

    # i = 0..n, page = current_document[i]
    for i, page in enumerate(current_document):

        result = classify_page(page)

        pages.append({"page_number": i + 1, "text": result["text"], "label": result["label"]})

    current_document.close()

    return pages



## actually chunking the text into paragraphs from the usable pages
## --> go through usable pages, extract text, split into paragraphs 
## for climateBERT; max token length = 512

def chunk_into_paragraphs(full_text):

    raw_chunks = re.split(r"\n{2,}", full_text)  ## split on 2 or more newlines

    paragraphs = []

    for chunk in raw_chunks:
        chunk = re.sub(r"\n", " ", chunk).strip() ## replace newlines with space and strip leading/trailing whitespace

        chunk = re.sub(r"\s+", " ", chunk)  

        chunk = re.sub(r"\.{3,}", " ", chunk) ## repace multiple dots with single space

        chunk = re.sub(r"^\d+\s+", "", chunk)  ## remove leading numbers 

        chunk = re.sub(r"(\w)-\s+(\w)", r"\1\2", chunk)  


        ## moved this from after running pipeline once
        chunk = re.sub(r"\xad\s*", "", chunk)
        chunk = re.sub(r"\x07\s*", "", chunk) # from 4th re-run
        ## from re-running, cgetting rid of unicode characters that arent useful
        chunk = "".join(c for c in chunk if unicodedata.category(c) != "Cc" or c in ["\n"])  ## remove control characters except newline and tab


        ## symbols can be found on google, but also if you look at the SR
        # reports some of them contain these; added in some other
        # that would commonly be used 
        chunk = re.sub(r"[•◦▪‣·►–—−]", " ", chunk) ## jot notes symbols + em-bases (from 2nd re-run)
        chunk = re.sub(r"SEE NOTE\s*\d+", " ", chunk, flags=re.IGNORECASE) # from 3rd rerun
        chunk = re.sub(r"SEE DECLARATION ON CORPORATE GOVERNANCE", " ", chunk, flags=re.IGNORECASE)

        chunk = re.sub(r"\s+", " ", chunk).strip()  


        word_count = len(chunk.split())

        if word_count < MIN_PARAGRAPH_LENGTH:
            continue

        if word_count <= MAX_PARAGRAPH_LENGTH:
            paragraphs.append(chunk)
        else:
            ## chunk is too long, split into smaller paragraphs
            sub_chunks = re.split(r"(?<=[.!?])\s+", chunk)  ## split on sentence boundaries

            ## from rerun: didn't account for sentences already exceeding the max length
            fixed_sub_chunks = []

            for s in sub_chunks:

                if len(s.split()) > MAX_PARAGRAPH_LENGTH:
                    ## split this sentence into smaller chunks
                    words = s.split()
                    for j in range(0, len(words), MAX_PARAGRAPH_LENGTH):
                        fixed_sub_chunks.append(" ".join(words[j:j + MAX_PARAGRAPH_LENGTH]))

                else:
                    fixed_sub_chunks.append(s)

            current, current_word_count = [], 0

            for sub_chunk in fixed_sub_chunks:
                sub_chunk_word_count = len(sub_chunk.split())

                if current_word_count + sub_chunk_word_count <= MAX_PARAGRAPH_LENGTH and current:
                    current.append(sub_chunk)
                    current_word_count += sub_chunk_word_count
                else:
                    if current and current_word_count >= MIN_PARAGRAPH_LENGTH:
                        paragraphs.append(" ".join(current))
                    current = [sub_chunk]
                    current_word_count = sub_chunk_word_count
            if current and current_word_count >= MIN_PARAGRAPH_LENGTH:
                paragraphs.append(" ".join(current))

    return paragraphs


def process_single_report(report_name, source_directory, metadata_row):


    pdf_path = os.path.join(source_directory, report_name)


    if not os.path.exists(pdf_path):
        print(f"ERROR. file not found: {report_name}")
        return []

    # print(f"current processing report: {report_name}")

    pages = extract_pages(pdf_path)

    ## joining all usable pages' text into a single string, separated by two newlines
    ## the newline separation for page turns 
    usable_texts = [p["text"] for p in pages if p["label"] == "usable"]
    full_text = "\n\n".join(usable_texts)

    paragraphs = chunk_into_paragraphs(full_text)

    ## re-creating the rows for the exploded dataframe, each row = 1 paragraph
    rows = []

    for i, p in enumerate(paragraphs):
        rows.append({
            "firm": metadata_row.get("firm", ""),
            "year": metadata_row.get("year", ""),
            "primary_sics_sector": metadata_row.get("primary_sics_sector", ""),
            "country": metadata_row.get("country", ""),
            "report_name": report_name,
            "paragraph_id": i,
            "text": p,
            "word_count": len(p.split())
        })

    return rows





## order of function calls:
# check_exploded_df_csv()
# process_single_report()
## ==> calls extract_pages() --> calls classify_page() --> calls get_usable_text()
## ==> then calls chunk_into_paragraphs()

# UPDATED for paraquet ver
if __name__ == "__main__":


    reports_folder = "/.../Sampled_Files/all_SR_reports/"

    exploded_df_file = "/.../Sampled_Files/final_df.csv"

    # output_csv_file = "/.../MRP/Sampled_Files/exploded_paragraphs.csv"
    output_parquet_file = "/.../Sampled_Files/Extracted_Paragraphs_Final/exploded_paragraphs.parquet"

    ## go through each pdf file in reports folder and extract paragraphs
    ## -_> using the exploded_df_file to get list of pdf files to process
    # --> the exploded_df_csv will be expanded into a new csv file with each row = 1 paragraph for that pdf file report

    exploded_df_csv = check_exploded_df_csv(exploded_df_file)

    all_rows = []
    missing_reports = []
    total = len(exploded_df_csv)
    start_time = time.time()

    for i, row in exploded_df_csv.iterrows():
        report_name = row["report_name"]
        print(f"[{i+1}/{total}] processing report: {report_name}")

        result_rows = process_single_report(report_name, reports_folder, row)

        if not result_rows:
            missing_reports.append(report_name)
            continue

        all_rows.extend(result_rows)

    duration = time.time() - start_time
    print(f"processing completed in {duration/60:.2f} minutes")
    print(f"reports skipped/missing: {len(missing_reports)}")

    df = pd.DataFrame(all_rows)
    # df.to_csv(output_csv_file, index=False)
    df.to_parquet(output_parquet_file, index=False)

    # print(f"saved {len(df)} paragraphs to {output_csv_file}")
    print(f"saved {len(df)} paragraphs to {output_parquet_file}")

    print("\n\n-----------------------------------------------------")
    print(f"by year:\n {df['year'].value_counts().sort_index()}")
    print(f"word count distribution:\n {df['word_count'].describe().round(1)}")
    print(f"\nsample paragraphs:\n {df['text'].sample(5).to_list()}")



## TO DO:
# check_exploded_df_csv()
# chunk_into_paragraphs()
# classify_page()
# extract_pages()
# process_single_report()
# the main block
