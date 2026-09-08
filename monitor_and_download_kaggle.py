r"""
Kaggle Training Monitor & Checkpoint Downloader
Monitors chamankanth/notebook27596bd2cf and downloads checkpoints to c:\Users\user\Downloads\checkpoint.
"""

import os
import sys
import time
import json
import subprocess

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, errors='replace')
    return result.stdout.strip(), result.stderr.strip(), result.returncode

def check_status(kernel_id="chamankanth/notebook27596bd2cf"):
    stdout, _, code = run_cmd(f"kaggle kernels status {kernel_id}")
    return stdout

def get_latest_logs(kernel_id="chamankanth/notebook27596bd2cf", max_lines=10):
    stdout, _, code = run_cmd(f"kaggle kernels logs {kernel_id}")
    lines = stdout.splitlines()
    step_lines = [l for l in lines if "Step " in l or "SUCCESS" in l or "Loss:" in l or "Model " in l]
    if step_lines:
        return "\n".join(step_lines[-max_lines:])
    return "\n".join(lines[-max_lines:]) if lines else "No log lines yet."

def download_outputs(kernel_id="chamankanth/notebook27596bd2cf", dest_dir=r"c:\Users\user\Downloads\checkpoint"):
    print(f"\n[+] Downloading checkpoints from Kaggle ({kernel_id}) into {dest_dir}...", flush=True)
    stdout, stderr, code = run_cmd(f'kaggle kernels output {kernel_id} -p "{dest_dir}"')
    print(stdout, flush=True)
    if stderr:
        print("Stderr:", stderr, flush=True)
    return code == 0

def monitor_loop(kernel_id="chamankanth/notebook27596bd2cf", dest_dir=r"c:\Users\user\Downloads\checkpoint", poll_interval=25):
    print("="*80, flush=True)
    print(f"      KAGGLE KERNEL MONITOR: {kernel_id}", flush=True)
    print("="*80, flush=True)
    
    start_time = time.time()
    last_reported_status = None
    
    while True:
        status_raw = check_status(kernel_id)
        elapsed = time.time() - start_time
        
        # Determine status
        is_running = "RUNNING" in status_raw
        is_complete = "COMPLETE" in status_raw
        is_error = "ERROR" in status_raw or "CANCELLED" in status_raw
        
        print(f"[{elapsed:.0f}s elapsed] Status: {status_raw}", flush=True)
        
        # Print latest logs
        logs = get_latest_logs(kernel_id, max_lines=5)
        if logs:
            print(f"--- Recent Logs ---\n{logs}\n-------------------", flush=True)
            
        if is_complete:
            print(f"\n[+] Kernel run {kernel_id} is COMPLETE! Initiating checkpoint download...", flush=True)
            success = download_outputs(kernel_id, dest_dir)
            if success:
                print(f"[+] All checkpoints successfully downloaded to {dest_dir}!", flush=True)
            else:
                print(f"[-] Download encountered an issue. Try running 'kaggle kernels output {kernel_id} -p {dest_dir}' manually.", flush=True)
            break
            
        if is_error:
            print(f"\n[-] Kernel encountered an error: {status_raw}", flush=True)
            print("Full logs:")
            full_logs, _, _ = run_cmd(f"kaggle kernels logs {kernel_id}")
            print(full_logs)
            break
            
        time.sleep(poll_interval)

if __name__ == '__main__':
    kernel_id = sys.argv[1] if len(sys.argv) > 1 else "chamankanth/notebook27596bd2cf"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else r"c:\Users\user\Downloads\checkpoint"
    monitor_loop(kernel_id, out_dir, poll_interval=30)
