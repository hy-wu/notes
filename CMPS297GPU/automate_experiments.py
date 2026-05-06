import os
import subprocess
import shutil
import time
import numpy as np

# --- Configuration ---
MODES = {
    0: "CLASSICAL",
    1: "RELATIVISTIC"
}
ORDERS = [0, 1, 2]
STEPS = 20000 
DT_VAL = 0.002
TESTPARTCL_VAL = 20 
OUTPUT_DIR = "experiment_results"
SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "./bamps_bin"

def run_command(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            line = output.decode().strip()
            if "Step" in line or "Performance" in line:
                print(line)
    return process.poll()

def automate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    summary_log = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    with open(summary_log, "w") as f:
        f.write("Mode,Order,DT,Steps,Avg_T,Avg_P_wall,Consistency_Error%\n")

    for m_val, m_name in MODES.items():
        for o in ORDERS:
            tag = f"{m_name}_ORDER{o}_DT{DT_VAL}_S{STEPS}"
            print(f"\n>>> STARTING EXPERIMENT: {tag} <<<")
            
            case_dir = os.path.join(OUTPUT_DIR, tag)
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)

            # 1. Compile with small DT and high STEPS
            compile_cmd = (f"nvcc -O3 -DMODE_RELATIVISTIC={m_val} -DENSKOG_ORDER={o} "
                           f"-DTOTAL_STEPS={STEPS} -DTESTPARTCL={TESTPARTCL_VAL} -DDELTA_T={DT_VAL} "
                           f"{SRC_FILE} -o {BIN_FILE}")
            
            if run_command(compile_cmd) != 0:
                print(f"Compilation failed for {tag}")
                continue

            # 2. Run Simulation
            if run_command(BIN_FILE) != 0:
                print(f"Execution failed for {tag}")
                continue

            # 3. Run Visualization
            if run_command("python3 viz_physics.py") != 0:
                print(f"Visualization failed for {tag}")

            # 4. Move and Organize Results
            files_to_move = {
                "physics_log.txt": f"{tag}_physics.log",
                "energies.txt": f"{tag}_energies.txt",
                "physics_validation.png": f"{tag}_validation.png",
                "spatial_distribution.png": f"{tag}_spatial.png"
            }
            
            for src, dst in files_to_move.items():
                if os.path.exists(src):
                    shutil.move(src, os.path.join(case_dir, dst))
            
            # 5. Cleanup
            if os.path.exists("positions.txt"): os.remove("positions.txt")
            
            # 6. Extract summary
            try:
                # Need to read the file we just moved
                log_path = os.path.join(case_dir, f"{tag}_physics.log")
                last_data = np.genfromtxt(log_path, skip_header=1)[-1]
                t_final = last_data[2]
                p_final = last_data[3] # Already scaled correctly in the .cu code now
                v = 1000.0
                n_phys = (last_data[5] / float(TESTPARTCL_VAL)) / v
                error = abs(p_final - n_phys * t_final) / (n_phys * t_final) * 100.0
                
                with open(summary_log, "a") as f:
                    f.write(f"{m_name},{o},{DT_VAL},{STEPS},{t_final:.4f},{p_final:.4f},{error:.2f}\n")
            except Exception as e:
                print(f"Summary extraction failed: {e}")

    print(f"\nAll experiments complete. Results organized in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    automate()
