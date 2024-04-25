import subprocess as sbp
import re
import time
import os

def check_job_status(job_id):
    """Check the status of the job until it's completed or encounters an error."""
    while True:
        # Execute the checkjob command
        result = sbp.run(['checkjob', job_id], capture_output=True, text=True)

        # Check for errors in executing checkjob
        if result.returncode != 0:
            print(f"Error checking job status: {result.stderr}")
            break

        # Parse the job state from the output
        match = re.search(r'State:\s+(\w+)', result.stdout)
        if match:
            state = match.group(1)
            print(f"Current job state: {state}")
            if state == 'Completed':
                print(f"Job {job_id} completed successfully.")
                break
            elif state in ['Idle', 'Running']:
                print(f"Job {job_id} is still {state.lower()}...")
                time.sleep(60)  # Check again after 60 seconds
            else:
                print(f"Job {job_id} encountered an unexpected state: {state}")
                break
        else:
            print("Could not determine job state from checkjob output.")
            break

def submit_job_and_get_id(submit_bash_script):
    """Submit a job with msub and return the job ID."""
    submit_command = ['msub', submit_bash_script]
    try:
        # Submit the job and capture the output
        #result = sbp.run(submit_command, check=True, capture_output=True, text=True)
        #result = sbp.run(submit_command, check=True, stdout=sbp.PIPE, stderr=sbp.PIPE, text=True)
        result = sbp.run(submit_command, check=True, stdout=sbp.PIPE, stderr=sbp.PIPE, universal_newlines=True)
        # The output is expected to be the job ID
        print(f"result.stdout: {result.stdout}")
        print(result)
        job_id = result.stdout.strip()
        return job_id
    except sbp.CalledProcessError as e:
        #print(f"Error submitting job: {e}")
        print(f"Error submitting job: {e.stderr}")
        #return None
        raise


def wait_for_file_creation(file_path, waiting_time,timeout=1800):
    start_time = time.time()  # Record the start time
    while True:
        if os.path.exists(file_path):
            print(f"The file '{file_path}' has been created.")
            return True
        elif (time.time() - start_time) > timeout:
            print(f"Timeout reached: The file '{file_path}' was not created within {timeout} seconds.")
            return False
        else:
            print(f"Waiting for the file '{file_path}' to be created. Checking again in 60 seconds...")
            time.sleep(waiting_time)  # Wait for 60 seconds before the next check

