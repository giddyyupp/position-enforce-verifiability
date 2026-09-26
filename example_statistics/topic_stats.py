import os
import json
from pathlib import Path
import numpy as np

MAIN_DIR = '.'
DATA_DIR = f'{MAIN_DIR}/merged_data_all_confs'

venues_all = os.listdir(DATA_DIR)


for venue in venues_all:    
    print(venue)
    papers = os.listdir(os.path.join(DATA_DIR, venue))

    topics = np.zeros(10, dtype=int)
    topics_codes = np.zeros(10, dtype=int)

    for paper in papers:

        try:
            paper_data = json.load(open(os.path.join(DATA_DIR, venue, paper), 'r'))
        except:
            paper_data = {}
            continue

        # topic stats
        if paper_data["topic_data"]:
            topic = paper_data["topic_data"]["results"][0]["topic"]["topic_id"]
            if topic:
                topics[topic-1] += 1
                # topic based code stats
                if paper_data["repo_data"] and paper_data["repo_data"]["assessment"]["training_code_available"]:
                    code_avail = paper_data["repo_data"]["assessment"]["training_code_available"]
                    if type(code_avail) == bool and code_avail:
                        topics_codes[topic-1] += 1
                    elif type(code_avail) == dict and code_avail["value"]:
                        topics_codes[topic-1] += 1

    
    for t in topics:
        print(t)
    print(topics)
    
    for t in topics_codes:
        print(t)
    print(topics_codes)



    