r"""
Kaggle Training Monitor & Checkpoint Downloader for Sparse-AST 500M
Monitors chamankanth/notebook27596bd2cf and downloads final_sparse_ast_500m.pt to c:\Users\user\Downloads\checkpoint.
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

def get_latest_logs(kernel_id="chamankanth/notebook27596bd2cf", max_lines=8):
    stdout, _, code = run_cmd(f"kaggle kernels logs {kernel_id}")
    lines = stdout.splitlines()
    step_lines = [l for l in lines if "Step " in l or "SUCCESS" in l or "Loss:" in l or "Model " in l or "GPU" in l]
    if step_lines:
        return "\n".join(step_lines[-max_lines:])
    return "\n".join(lines[-max_lines:]) if lines else "No log lines yet."

def download_outputs(kernel_id="chamankanth/notebook27596bd2cf", dest_dir=r"c:\Users\user\Downloads\checkpoint", pattern="final_sparse_ast_500m.pt"):
    print(f"\n[+] Downloading final 500M checkpoint from Kaggle ({kernel_id}) into {dest_dir}...", flush=True)
    cmd = f'kaggle kernels output {kernel_id} -p "{dest_dir}"'
    if pattern:
        cmd += f' --file-pattern "{pattern}"'
    stdout, stderr, code = run_cmd(cmd)
    print(stdout, flush=True)
    if stderr:
        print("Stderr:", stderr, flush=True)
    return code == 0

def monitor_loop(kernel_id="chamankanth/notebook27596bd2cf", dest_dir=r"c:\Users\user\Downloads\checkpoint", poll_interval=30):
    print("="*80, flush=True)
    print(f"      KAGGLE KERNEL MONITOR: {kernel_id} (Sparse-AST 500M)", flush=True)
    print("="*80, flush=True)
    
    start_time = time.time()
    
    while True:
        status_raw = check_status(kernel_id)
        elapsed = time.time() - start_time
        
        is_running = "RUNNING" in status_raw
        is_complete = "COMPLETE" in status_raw
        is_error = "ERROR" in status_raw or "CANCELLED" in status_raw
        
        print(f"[{elapsed:.0f}s elapsed] Status: {status_raw}", flush=True)
        
        logs = get_latest_logs(kernel_id, max_lines=6)
        if logs:
            print(f"--- Recent Logs ---\n{logs}\n-------------------", flush=True)
            
        if is_complete:
            print(f"\n[+] Kernel run {kernel_id} is COMPLETE! Initiating checkpoint download...", flush=True)
            success = download_outputs(kernel_id, dest_dir, pattern="final_sparse_ast_500m.pt")
            if success:
                print(f"[+] Final 500M checkpoint successfully downloaded to {dest_dir}!", flush=True)
            else:
                print(f"[-] Download encountered an issue. Retrying full download...", flush=True)
                download_outputs(kernel_id, dest_dir, pattern=None)
            break
            
        if is_error:
            print(f"\n[-] Kernel encountered an error: {status_raw}", flush=True)
            logs_full = get_latest_logs(kernel_id, max_lines=25)
            print(f"Full Logs:\n{logs_full}", flush=True)
            break
            
        time.sleep(poll_interval)

if __name__ == '__main__':
    kernel_id = sys.argv[1] if len(sys.argv) > 1 else "chamankanth/notebook27596bd2cf"
    dest = sys.argv[2] if len(sys.argv) > 2 else r"c:\Users\user\Downloads\checkpoint"
    monitor_loop(kernel_id, dest)
