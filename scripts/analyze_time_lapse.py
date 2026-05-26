import json
import os
from pathlib import Path
from datetime import datetime

def analyze_time_lapse(folder_path):
    root = Path(folder_path)
    iter_files = sorted(list(root.glob("*_iter_*.json")))
    
    # Group files by experiment
    experiments = {}
    for f in iter_files:
        exp_id = "_".join(f.name.split("_")[:3])
        if exp_id not in experiments:
            experiments[exp_id] = []
        experiments[exp_id].append(f)
    
    print(f"### ITERATION TIME-LAPSE ANALYSIS: {folder_path} ###\n")
    
    total_duration_all = 0
    
    for exp_id, files in sorted(experiments.items()):
        print(f"#### Experiment: {exp_id}")
        print("| Iter | Duration | RD Cost | Cumul. Time | Timestamp |")
        print("|---|---|---|---|---|")
        
        prev_mtime = None
        start_mtime = None
        cumul_time = 0
        
        # Sort files by iteration number
        files.sort(key=lambda x: int(x.name.split("_")[-1].split(".")[0]))
        
        for f in files:
            mtime = os.path.getmtime(f)
            if start_mtime is None:
                start_mtime = mtime
            
            duration = 0
            if prev_mtime is not None:
                duration = mtime - prev_mtime
                cumul_time = mtime - start_mtime
            
            with open(f, 'r') as jf:
                try:
                    data = json.load(jf)
                    cost = data.get('total_cost')
                    cost_str = f"{cost:,.2f}" if cost is not None else "N/A"
                except:
                    cost_str = "Error"
            
            iter_num = int(f.name.split("_")[-1].split(".")[0])
            ts_str = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
            
            dur_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}" if duration > 0 else "00:00"
            cum_str = f"{int(cumul_time // 60):02d}:{int(cumul_time % 60):02d}"
            
            print(f"| {iter_num:3d} | {dur_str} | {cost_str:>12s} | {cum_str} | {ts_str} |")
            prev_mtime = mtime
            
        exp_duration = prev_mtime - start_mtime
        total_duration_all += exp_duration
        print(f"\n**Total Experiment Duration:** {int(exp_duration // 60)} min {int(exp_duration % 60)} sec\n")

    print(f"**Total Suite Execution Time:** {int(total_duration_all // 3600)}h {int((total_duration_all % 3600) // 60)}m")

if __name__ == "__main__":
    analyze_time_lapse("fast_validated_sweep_20260524_235753")
