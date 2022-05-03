import os
import glob
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

from gimme_all_results_sets_lib import extract_results 

# Create the jinja2 environment.
current_directory = os.path.dirname(os.path.abspath(__file__))
env = Environment(loader=FileSystemLoader(current_directory))

# Find all files with the j2 extension in the current directory
templates = glob.glob('*.j2') 

# What all do we want part of our overview?:

# 1 - Global tag name e.g.: conformer
# 2 - Subtag name

consider_out_dirs = ["conformer/best/"] # TODO: use should be asked which of these he wants updated
consider_out_dirs = ["conformer/se_block/"]

# For now consider all!
# TODO; uncomment
consider_out_dirs = glob.glob("alias/conformer/*")
consider_out_dirs = [x.replace("alias/", "") + "/" for x in consider_out_dirs]

print()
print("====================================================")
print("= Checking setup status " + str(datetime.now()) + " =")
print("====================================================")
print("Considering: " + str(consider_out_dirs))

update = [True] * len(consider_out_dirs) # Same as above, for now we want to consider all!
_dir_files = {}
_first_names = []
# Load this from alreay known
_file_data_map = {}


def filt(name): # Removes the dataset tag from experiment name
    for x in ["dev-other", "dev-clean", "test-other", "test-clean"]:
        if x in name:
            return name[-(len(x)+1)]
    else:
        return name

i = 0
for drr in consider_out_dirs:
    _first_names.append(drr)
    if drr not in _dir_files:
        #print("DIR: " + drr)
        _dir_files[drr] = glob.glob("alias/" + drr + "*") # gets all them files in that dir

        # Now they need to be filtered
        _dir_files[drr] = [f for f in _dir_files[drr] if not ("recog_" in f) ]

        # scan in the files
        for file in _dir_files[drr]:

            filtered_name = filt(file)
            filtered_name = filtered_name.split("/")[-1]
            if filtered_name not in _file_data_map:
                if update[i]:
                    print("Processing: " + filtered_name)
                    try:
                        _file_data_map[filtered_name] = extract_results(drr[:-1], [filtered_name])
                    except Exception as e:
                        _file_data_map[filtered_name] = str(e)
                        print(e)
                #_file_data_map[filtered_name] = "NOT_GENERATED"


        # remove unwanted indexes from files
        for tag in ["dev-other", "dev-clean", "test-other", "test-clean"]:
            if tag in drr:
                del _dir_files[tag] # this delete only the element?

    i += 1


_dir_files_short_names = { name : sorted([ d.split("/")[-1] for d in _dir_files[name] ]) for name in _dir_files.keys()} # Short name descriptions

import re

# I know this whole thing is computation hell, but only used once in a while anyways
def content_map_filter_data(map):
    map_list = map.split("\n")
    # filters all epochs and epoch data of the content map
    ex_float_or_int = r'[\d\.\d]+' #r"[-+]?(?:\d*\.\d+|\d+)" <- might be a bit more precise
    wers_by_dataset_epoch = {}
    best_wer_for_set = {}
    best_ep_dev_other = -1
    for tag in ["dev-other", "dev-clean", "test-other", "test-clean"]:
        wers_by_dataset_epoch[tag] = {}
        data = [k for k in map_list if ("WER" in k) and (tag in k)]
        epoch_data = [k for k in map_list if ("epoch" in k and "~" in k)]

        cur_best = 999.0
        best_ep = -1
        if len(data) == 0:
            best_wer_for_set[tag] = f"NONE"
            continue

        for i in range(len(data)):
            #print("data", data[i])
            #print("ep", epoch_data[i])
            WERS = re.findall(ex_float_or_int, data[i]) # First is non tuned LM
            EP = re.findall(ex_float_or_int, epoch_data[i]) # First is epoch number, second is time ( TODO store also time? )
            #print("EPS", EP)
            #print("WER", WERS)
            if float(WERS[-1]) < cur_best:
                cur_best = float(WERS[-1]) # Doing the convertion over and over :P
                best_ep = int(EP[0])
                if tag == "dev-other":
                    best_ep_dev_other = int(EP[0])
            wers_by_dataset_epoch[tag][int(EP[0])] = [WERS[0], WERS[-1]] # The one in the middle is '4'gramLM

        best_wer_for_set[tag] = f"ep: {best_ep}, WER: {cur_best}"

    #print(wers_by_dataset_epoch)
    #print(best_wer_for_set)
    # Now we got the best wer for evry tag but acutally we always only want to display the best for dev-other, and then same epoch for the other datasets

    for tag in ["dev-clean", "test-other", "test-clean"]:
        if best_ep_dev_other in wers_by_dataset_epoch[tag]:
            best_wer_for_set[tag] = f"ep: {best_ep_dev_other}, WER: {wers_by_dataset_epoch[tag][best_ep_dev_other]}"
        else:
            best_wer_for_set[tag] = f"no data for ep {best_ep_dev_other}"

    return best_wer_for_set

#print(_dir_files_short_names)

first_names = consider_out_dirs
second_names = _dir_files_short_names
content_map = _file_data_map

summary_file = True
generate_csv = True # Csv files with only the best WER

csv_columns = {
    "NAME" : [],

    "BEST WER (dev-other)" : [],
    "WER (dev-clean)" : [],

    "WER (test-other)": [],
    "WER (test-clean)": [],

    "FULL CONFIG PATH" : []
}

if summary_file:
    # Also generate a summary file for all results
    for x in _dir_files_short_names:
        sub_dir_name = "results/" + x.replace("conformer/","")
        if not os.path.exists(sub_dir_name):
            os.mkdir(sub_dir_name)

        # Short summary
        short_sum = "" # Every sub setup also generates a 'short summary'


        if generate_csv:
            l = [x,"-","-","-", "-", "-"]
            keys = list(csv_columns.keys())
            for i in range(len(keys)):
                csv_columns[keys[i]].append(l[i])

        # This generates all individual summary files
        for exp in _dir_files_short_names[x]:
            file_name = sub_dir_name + exp
            outp = open(file_name, "w")
            n = outp.write(content_map[exp])

            if generate_csv: # Make this if generate_csv
                best_wer_by_set = content_map_filter_data(content_map[exp])
                full_config_path = os.getcwd() + "/output/" + x + exp + "/returnn.config"
                l = [exp] + [best_wer_by_set[data] for data in ["dev-other", "dev-clean", "test-other", "test-clean"]] + [full_config_path]
                keys = list(csv_columns.keys())
                for i in range(len(keys)):
                    csv_columns[keys[i]].append(l[i])
                    

            # The short summary only accounts for dev-other and the most recent epoch
            data = [k for k in content_map[exp].split("\n") if ("WER" in k) and ("dev-other" in k)]
            epoch_data = [k for k in content_map[exp].split("\n") if ("epoch" in k)]
            if len(data) == 0:
                short_sum += f"{exp}: no wer found \n\n"
            else:
                short_sum += f"{exp}: {data[-1]}   | {epoch_data[-1]} | \n\n"
                # short_sum just shows the last ep, we want to store the best though TODO 
            outp.close()

        # Write the summary file
        outp = open(sub_dir_name + "summary.txt", "w")
        n = outp.write(short_sum)
        outp.close()

    if generate_csv:
        print(csv_columns)
        import csv
        f = open('summary.csv', 'w')

        writer = csv.writer(f)
        keys = list(csv_columns.keys())
        writer.writerow(keys)

        for x in range(len(csv_columns[keys[0]])):
            print(x)
            writer.writerow([csv_columns[k][x] for k in keys])

        f.close()

    



        
            
# refove all recogs
#print(_file_data_map)


def render_template(filename, f, s, c):
    return env.get_template(filename).render(
        test="test",
        first_names = f,
        second_map = s,
        contentmap = c
    )

rendered = render_template("experiments_overview.j2", first_names, second_names, content_map )

if True:
    outp = open("overviews/overview.html", "w")
    n = outp.write(rendered)
    outp.close()
else:
    print(rendered)
