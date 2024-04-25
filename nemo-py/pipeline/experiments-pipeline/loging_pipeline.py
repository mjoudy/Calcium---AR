import os
from datetime import datetime
import csv
import json

# Specify the file name here
file_name = "pipeline_log.txt"

def check_file_exists(file_name):
    return os.path.exists(file_name)

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d --- %H:%M:%S")

def is_last_line_today_date(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].split()[0]
            return last_line == get_current_date().split()[0]
        else:
            return False

def is_last_line_dashes(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
        if lines:
            last_line = lines[-1].strip()
            return last_line == "-" * 50
        else:
            return False


def append_line_to_file(file_name, line):
    with open(file_name, "a") as file:
        file.write(line + "\n")

def log_process(custom_message="Your custom message here"):
    if check_file_exists(file_name):
        if is_last_line_today_date(file_name):
            append_line_to_file(file_name, f"{get_current_date()}: {custom_message}")
        elif is_last_line_dashes(file_name):
            append_line_to_file(file_name, f"{get_current_date()}: {custom_message}")
        else:
            temp_text = "\n" + "-" * 50
            append_line_to_file(file_name, temp_text)
    else:
        with open(file_name, "w") as file:
            file.write(get_current_date() + ": " + custom_message)


def write_dict_to_json(data_dict, filename):
    with open(filename, 'w') as f:
        json.dump(data_dict, f)

def add_results_to_json(result_key, result_value, filename):
    with open(filename, 'r+') as f:
        data_dict_new = json.load(f)
        data_dict_new[result_key] = result_value
        f.seek(0)
        json.dump(data_dict_new, f)
        f.truncate()

def append_dict_to_csv(json_name, csv_name):
    with open(json_name, 'r') as f:
        data_dict_final = json.load(f)

    field_names = list(data_dict_final.keys())

    with open(csv_name, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        if csvfile.tell() == 0:  # If file is empty, write header
            writer.writeheader()
        writer.writerow(data_dict_final)