import os
import subprocess
import shutil
import time

# --- Configuration ---
MODES = {
    0: "CLASSICAL",
    1: "RELATIVISTIC"
}
ORDERS = [0, 1, 2]
STEPS = 3000 # Reduced to avoid timeout after 64-color upgrade
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
            if "Step" in line or "Performance" in line: # Only show progress
                print(line)
    return process.poll()

def automate():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    summary_log = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    with open(summary_log, "w") as f:
        f.write("Mode,Order,Avg_T,Avg_P_wall,Consistency_Error%\n")

    for m_val, m_name in MODES.items():
        for o in ORDERS:
            tag = f"{m_name}_ORDER{o}"
            print(f"\n>>> STARTING EXPERIMENT: {tag} ({STEPS} steps) <<<")
            
            case_dir = os.path.join(OUTPUT_DIR, tag)
            if not os.path.exists(case_dir):
                os.makedirs(case_dir)

            # 1. Compile with specific flags including TOTAL_STEPS
            compile_cmd = f"nvcc -O3 -DMODE_RELATIVISTIC={m_val} -DENSKOG_ORDER={o} -DTOTAL_STEPS={STEPS} {SRC_FILE} -o {BIN_FILE}"
            if run_command(compile_cmd) != 0:
                print(f"Compilation failed for {tag}")
                continue

            # 2. Run Simulation
            if run_command(BIN_FILE) != 0:
                print(f"Execution failed for {tag}")
                continue

            # 3. Run Visualization
            # Note: viz_physics.py depends on bamps_master.log, energies.txt, physics_log.txt
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
            if os.path.exists("refined_analysis.png"): os.remove("refined_analysis.png")
            try:
                # Read last line of current physics_log
                last_data = np.genfromtxt(os.path.join(case_dir, f"{tag}_physics.log"), skip_header=1)[-1]
                # stats format in file: Step Time Temp Pressure_Wall Energy Count
                t_final = last_data[2]
                p_final = last_data[3] / 100.0 # Re-applying the 1/testpartcl scaling
                v = 1000.0
                n_phys = (last_data[5] / 100.0) / v
                error = abs(p_final - n_phys * t_final) / (n_phys * t_final) * 100.0
                
                with open(summary_log, "a") as f:
                    f.write(f"{m_name},{o},{t_final:.4f},{p_final:.4f},{error:.2f}\n")
            except:
                pass

    print(f"\nAll experiments complete. Results organized in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    import numpy as np # needed for summary extraction
    automate()
