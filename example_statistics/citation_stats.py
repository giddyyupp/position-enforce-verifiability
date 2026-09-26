import os
import json
from pathlib import Path
import numpy as np


def check_code_avail(paper_data):
    code_avail_ret = False
    # topic based code stats
    if paper_data["repo_data"] and paper_data["repo_data"]["assessment"]["training_code_available"]:
        code_avail_test = paper_data["repo_data"]["assessment"]["inference_or_test_code_available"]
        code_avail_train = paper_data["repo_data"]["assessment"]["training_code_available"]
        if type(code_avail_test) == bool and type(code_avail_train) == bool and (code_avail_test or code_avail_train):
            code_avail_ret = True
        elif type(code_avail_test) == dict and type(code_avail_train) == dict and (code_avail_test["value"] or code_avail_train["value"]):
            code_avail_ret = True
        else:
            pass

    return code_avail_ret

MAIN_DIR = '.'
DATA_DIR = f'{MAIN_DIR}/merged_data_all_confs'

YEARS = [2025, 2024, 2023, 2022, 2021]
citations_d = {}
# dict building
for year in YEARS:
    citations_d[year] = {'code': [], 'nocode': [], 'all': []}

venues_all = os.listdir(DATA_DIR)

venues_all = sorted(venues_all)

for venue in venues_all:    
    print(venue)
    year = int(venue.split('_')[1])
    papers = os.listdir(os.path.join(DATA_DIR, venue))

    papers = sorted(papers)

    for paper in papers:

        try:
            paper_data = json.load(open(os.path.join(DATA_DIR, venue, paper), 'r'))
        except:
            paper_data = {}
            continue

        # citation counts
        if paper_data["citation"]:
            citation_count = paper_data["citation"]["citationCount"]
        
        # maybe exclude 0 citation papers?
        # if citation_count == 0:
        #     continue

        code_avail = check_code_avail(paper_data)
        if code_avail:
            citations_d[year]['code'].append(citation_count)
        else:
            citations_d[year]['nocode'].append(citation_count)    

        citations_d[year]['all'].append(citation_count)    

print(f"CODE AVAIL MEAN:")
for y in YEARS:
    print(np.mean(citations_d[y]['code']))

print(f"CODE NOT AVAIL MEAN:")
for y in YEARS:
    print(np.mean(citations_d[y]['nocode']))

print(f"ALL PAPER MEAN:")
for y in YEARS:
    print(np.mean(citations_d[y]['all']))

    