import subprocess
import sys

model_name = sys.argv[1]
transpose = sys.argv[2]
input_path = "audios/tt3.mp3"
index_path = f"logs/{model_name}/trained_IVF1203_Flat_nprobe_1_{model_name}_v2.index"
f0_method = "rmvpe"
opt_path = f"audios/{model_name}.wav"
model_name = f"{model_name}.pth"
index_rate = 0
volume_normalization = 0
consonant_protection = 0

command = [
    "python3",
    "tools/infer_cli.py",
    "--f0up_key", str(transpose),
    "--input_path", input_path,
    "--index_path", index_path,
    "--f0method", f0_method,
    "--opt_path", opt_path,
    "--model_name", model_name,
    "--index_rate", str(index_rate),
    "--device", "cuda:0",
    "--is_half", "True",
    "--filter_radius", "3",
    "--resample_sr", "0",
    "--rms_mix_rate", str(volume_normalization),
    "--protect", str(consonant_protection)
]

try:
    subprocess.run(command, check=True)
    print("Inference completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during inference: {e}")

