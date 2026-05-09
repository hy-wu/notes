import os
import subprocess
import shutil
import time
import numpy as np

# --- Ultimate Precision Configuration ---
MODES = { 0: "CLASSICAL" }
ORDERS = [0] 
STEPS = 20000 
DT_VAL = 0.002 
TESTPARTCL_VAL = 1 
SIGMA_VAL = 0.2    # Significantly reduced as requested
EPSILON_VAL = 0.01 # Small epsilon for weak interaction
N_TARGET = 10000   # Increased for better statistics

OUTPUT_DIR = "experiment_results"
SRC_FILE = "code/bamps_gpu_ancient.cu"
BIN_FILE = "./bamps_bin"

def run_command(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None: break
        if output:
            line = output.decode().strip()
            if "Step" in line or "Performance" in line: print(line)
    return process.poll()

def automate():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    summary_log = os.path.join(OUTPUT_DIR, "experiment_summary.csv")
    with open(summary_log, "w") as f:
        f.write("Mode,LJ,N,Sigma,Avg_T,P_wall,P_virial,P_vdW,Error%\n")

    for lj_val in [0, 1]: # Compare Ideal vs LJ
        tag = f"CLASSICAL_LJ{lj_val}_FINAL_N{N_TARGET}"
        print(f"\n>>> STARTING ULTIMATE PRECISION: {tag} <<<")
        case_dir = os.path.join(OUTPUT_DIR, tag)
        if not os.path.exists(case_dir): os.makedirs(case_dir)

        compile_cmd = (f"nvcc -O3 -DMODE_RELATIVISTIC=0 -DMODE_LJ={lj_val} -DENSKOG_ORDER=0 "
                       f"-DTOTAL_STEPS={STEPS} -DTESTPARTCL={TESTPARTCL_VAL} -DDELTA_T={DT_VAL} "
                       f"-DSIGMA_OVERRIDE={SIGMA_VAL} -DEPSILON_OVERRIDE={EPSILON_VAL} "
                       f"-DN_PARTICLES_OVERRIDE={N_TARGET} "
                       f"{SRC_FILE} -o {BIN_FILE}")
        
        if run_command(compile_cmd) != 0: continue
        if run_command(BIN_FILE) != 0: continue
        if run_command("python3 viz_physics.py") != 0: continue

        files_to_move = {
            "physics_log.txt": f"{tag}_physics.log",
            "energies.txt": f"{tag}_energies.txt",
            "physics_validation.png": f"{tag}_validation.png",
            "spatial_distribution.png": f"{tag}_spatial.png"
        }
        for src, dst in files_to_move.items():
            if os.path.exists(src): shutil.move(src, os.path.join(case_dir, dst))
        
        try:
            log_path = os.path.join(case_dir, f"{tag}_physics.log")
            last_data = np.genfromtxt(log_path, skip_header=1)[-1]
            t_f = last_data[2]; p_w = last_data[3]; p_v = last_data[6]
            v = 1000.0; n_dens = (last_data[5] / float(TESTPARTCL_VAL)) / v
            b = (2.0/3.0) * np.pi * (SIGMA_VAL**3)
            a = (16.0/9.0) * np.pi * EPSILON_VAL * (SIGMA_VAL**3)
            p_theory = (n_dens * t_f) / (1.0 - n_dens * b) - a * (n_dens**2) if lj_val else n_dens*t_f
            err = abs(p_v - p_theory) / p_theory * 100.0
            
            with open(summary_log, "a") as f:
                f.write(f"CLASSICAL,{lj_val},{N_TARGET},{SIGMA_VAL},{t_f:.4f},{p_w:.4f},{p_v:.4f},{p_theory:.4f},{err:.2f}\n")
        except: pass

    print(f"\nFinal Validation complete. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    automate()
