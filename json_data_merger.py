"""
Docstring for json_data_merger
Merging the various information of each paper into a single json to easier analysis.
"""

import os
import json
from pathlib import Path

MIN_YEAR = 2021

MAIN_DIR = './'

PDFS_DIR = f'{MAIN_DIR}/all_pdfs'
OUT_DIR = f'{MAIN_DIR}/merged_data_all_confs'

MATCHED_DIR = f'{MAIN_DIR}/matched_json'
CITATION_DIR = f'{MAIN_DIR}/citation_counts'
REPO_CHECK_DIR = f'{MAIN_DIR}/repos_checked'
TOPIC_DIR = f'{MAIN_DIR}/topic_checker'
URL_CHECKER_DIR = f'{MAIN_DIR}/url_checker'

os.makedirs(OUT_DIR, exist_ok=True)

venues_all = os.listdir(PDFS_DIR)

venues = []

# filter out years before 2021
for venue in venues_all:
    if int(venue.split('_')[1]) >= 2021:
        venues.append(venue)

for venue in venues:
    os.makedirs(os.path.join(OUT_DIR, venue), exist_ok=True)
    print(venue)
    
    # papers = os.listdir(os.path.join(PDFS_DIR, venue))
    try:
        match_info = json.load(open(os.path.join(MATCHED_DIR, venue + '.json'), 'r'))
    except:
        match_info = {}
    
    for mapped_data in match_info['mapped']:
        pdf_name = mapped_data['pdf']
        paper_json = os.path.splitext(pdf_name)[0] + '.json'
        
        # read all relevant jsons if exist, else put empty dict for missing.
        try:
            citation_info = json.load(open(os.path.join(CITATION_DIR, venue, paper_json), 'r'))
        except:
            citation_info = {}
        try:
            repo_info = json.load(open(os.path.join(REPO_CHECK_DIR, venue, paper_json), 'r'))
        except:
            repo_info = {}
        try:
            topic_info = json.load(open(os.path.join(TOPIC_DIR, venue, paper_json), 'r'))
        except:
            topic_info = {}
        try:
            url_info = json.load(open(os.path.join(URL_CHECKER_DIR, venue, paper_json), 'r'))
        except:
            url_info = {}

        merged_data = {"metadata": mapped_data, 
                       "citation": citation_info, 
                       "repo_data": repo_info,
                       "topic_data": topic_info,
                       "url_data": url_info}
        
        Path(os.path.join(OUT_DIR, venue, paper_json)).write_text(json.dumps(merged_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved: {os.path.join(OUT_DIR, venue, paper_json)}")

        print('DONE!')



