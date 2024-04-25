import os
from datetime import datetime

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