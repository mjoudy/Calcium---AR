from datetime import datetime as dtm

def print_time(text, file_name, overwrite = False):
   
    if overwrite == True:
        current_time = dtm.now()
        time_str = current_time.strftime("%H:%M:%S")
        with open(file_name, 'w') as f:
            f.write(text + time_str + '\n')

    else:
        current_time = dtm.now()
        time_str = current_time.strftime("%H:%M:%S")
        with open(file_name, 'a') as f:
            f.write(text + time_str + '\n')


