import os
import json

input_dir = './repos_checked'

venues = os.listdir(input_dir)
venues = sorted(venues)

for venue in venues:
    print(venue)

    json_files = os.listdir(os.path.join(input_dir, venue))

    valid_train = 0
    valid_inference = 0
    weights_available = 0
    ack_present = 0
    issues_present = 0
    
    valid_train_no_infer = 0
    valid_infer_no_train = 0
    both_valid = 0

    for json_file in json_files:
        data = json.load(open(os.path.join(input_dir, venue, json_file), 'r'))

        try:
            train_code_avail = data['assessment']['training_code_available']["value"]
            if train_code_avail:
                valid_train += 1
        except:
            train_code_avail = False
        try:
            infer_code_avail = data['assessment']['inference_or_test_code_available']["value"]
            if infer_code_avail:
                valid_inference += 1
        except:
            infer_code_avail = False

        try:
            weights_avail = data['assessment']['weights_available_for_this_method']["value"]
            if weights_avail:
                weights_available += 1
        except:
            weights_avail = False
        
        try:
            ack_avail = data['assessment']['acknowledgement']["value"]
            if ack_avail:
                ack_present += 1
        except:
            ack_avail = False
        try:
            issues_avail = data['assessment']['open_issues_about_installation_or_reproducibility']["value"]
            if issues_avail:
                issues_present += 1
        except:
            issues_avail = False

        if train_code_avail and infer_code_avail:
            both_valid += 1
        elif train_code_avail and not infer_code_avail:
            valid_train_no_infer += 1
        elif not train_code_avail and infer_code_avail:
            valid_infer_no_train += 1



    print(valid_train)
    print(valid_inference)
    print(weights_available)
    print(ack_present)
    print(issues_present)

    print(both_valid)
    print(valid_train_no_infer)
    print(valid_infer_no_train)
