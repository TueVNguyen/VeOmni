import subprocess

runs = []
gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]
for i, gpu_id in enumerate(gpu_ids):
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python infer_logit.py {i}"
    run = subprocess.Popen(command, shell=True)
    runs.append(run)
    
for run in runs:
    run.wait()