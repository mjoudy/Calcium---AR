from datetime import datetime as dtm

def print_time(filename='log.txt', text='current tiom', new='w'):
    current_time = dtm.now()
    time_str = current_time.strftime("%H:%M:%S")
    with open(filename, new) as f:
        f.write(text + time_str + '\n')

def half_chaunk(signal_file=)
