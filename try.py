import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from matplotlib.backends.backend_pdf import PdfPages
from signal_utils import preprocess_pcg
from tqdm import tqdm
# --- Envelope variance using Hilbert envelope ---
def envelope_variance(signal):
    analytic = librosa.effects.hilbert(signal)
    envelope = np.abs(analytic)
    return np.var(envelope)
# --- Load metadata CSV ---
csv_path = "/kaggle/input/circor-and-physionet/Data_Circor_and_Physionet_FInal.csv"
df = pd.read_csv(csv_path)
# Filter only Normal samples
df_normal = df[df['condition'] == 'Normal']
# Get best 1000 based on spectral flatness (lowest = best quality)
df_normal_sorted = df_normal.sort_values('spectral_flatness', ascending=True).head(1000)
# Process each to get center 6s envelope variance
envelope_vars = []
segment_info = []
for row in tqdm(df_normal_sorted.itertuples(), desc="Processing center 6s segments"):
    try:
        audio_path = row.recording_path
        source_fs = int(row.source_fs)
        # Load full signal
        full_signal, fs = librosa.load(audio_path, sr=source_fs)
        # Preprocess
        pcg = preprocess_pcg(full_signal, original_fs=fs, resample_fs=1450, band=(20, 720))
        total_len = len(pcg)
        total_duration = total_len / 1450
        if total_duration > 6.0:
            center = total_len // 2
            half_len = 3 * 1450
            start = max(0, center - half_len)
            end = min(total_len, center + half_len)
            segment = pcg[start:end]
            start_time = start / 1450
            end_time = end / 1450
        else:
            segment = pcg
            start_time = 0.0
            end_time = total_duration
        env_var = envelope_variance(segment)
        envelope_vars.append(env_var)
        segment_info.append({
            "recording_path": audio_path,
            "source_fs": fs,
            "start_time": start_time,
            "end_time": end_time,
            "envelope_variance": env_var
        })
    except Exception as e:
        envelope_vars.append(-np.inf)
        segment_info.append({
            "recording_path": row.recording_path,
            "source_fs": None,
            "start_time": None,
            "end_time": None,
            "envelope_variance": -np.inf
        })
# Add results to dataframe
df_normal_sorted = df_normal_sorted.copy()
df_normal_sorted["envelope_variance"] = envelope_vars
df_normal_sorted["center_start_time"] = [info["start_time"] for info in segment_info]
df_normal_sorted["center_end_time"] = [info["end_time"] for info in segment_info]
# Select top 500 by envelope variance
df_best = df_normal_sorted.sort_values('envelope_variance', ascending=False).head(500)
# Save to CSV
df_best.to_csv("best_500_normal_pcg_center6sec.csv", index=False)
print("CSV saved: best_500_normal_pcg_center6sec.csv")
# Plot and save to PDF
with PdfPages('best_500_normal_pcg_center6sec.pdf') as pdf:
    for i, row in tqdm(df_best.iterrows(), total=len(df_best), desc="Plotting center 6s PCGs"):
        audio_path = row['recording_path']
        source_fs = int(row['source_fs']) if pd.notna(row['source_fs']) else None
        start_time = row['center_start_time']
        end_time = row['center_end_time']
        try:
            if source_fs is None:
                raise ValueError("Missing sampling rate")
            full_signal, fs = librosa.load(audio_path, sr=source_fs)
            pcg = preprocess_pcg(full_signal, original_fs=fs, resample_fs=1450, band=(20, 720))
            total_len = len(pcg)
            total_duration = total_len / 1450
            if total_duration > 6.0:
                center = total_len // 2
                half_len = 3 * 1450
                start = max(0, center - half_len)
                end = min(total_len, center + half_len)
                segment = pcg[start:end]
                time_axis = np.arange(len(segment)) / 1450
                segment_str = f"{start/1450:.2f}-{end/1450:.2f}s"
            else:
                segment = pcg
                time_axis = np.arange(len(segment)) / 1450
                segment_str = f"0.00-{total_duration:.2f}s (short)"
            plt.figure(figsize=(10, 3))
            plt.plot(time_axis, segment)
            plt.title(
                f"Normal Sample {i+1} | Flatness: {row['spectral_flatness']:.4f} | Envelope Var: {row['envelope_variance']:.2f}\n"
                f"Recording: {audio_path}\nSegment: {segment_str}"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        except Exception as e:
            print(f"Error plotting {audio_path}: {e}")
print("PDF saved: best_500_normal_pcg_center6sec.pdf")