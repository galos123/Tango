import os
import time
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Stability settings
os.environ["HUGGINGFACE_HUB_TIMEOUT"] = "120"


def download_with_retry(repo_id, local_dir, allow_patterns, max_retries=10):
    attempt = 0
    while attempt < max_retries:
        try:
            print(f"📥 Download Attempt {attempt + 1}/{max_retries}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=1,
                resume_download=True,
            )
            print("✅ Download successful!")
            return True
        except Exception as e:
            attempt += 1
            wait_time = min(attempt * 30, 300)
            print(f"⚠️ Download interrupted: {e}")
            if attempt < max_retries:
                print(f"🔄 Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                return False


def download_and_convert_to_wav(target_dir="./original_data"):
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    temp_dir = Path("./temp_download")

    # --- PHASE 1: DOWNLOAD ---
    if not download_with_retry(
        repo_id="cvssp/WavCaps",
        local_dir=temp_dir,
        allow_patterns=[
            "Zip_files/AudioSet_SL/AudioSet_SL.z*",
            "json_files/AudioSet_SL/*.json",
        ],
    ):
        return

    # --- PHASE 2: EXTRACTION (Using 7z) ---
    main_zip = temp_dir / "Zip_files" / "AudioSet_SL" / "AudioSet_SL.zip"

    if not (target_path / "AudioSet_SL").exists():
        print("📦 Extracting multi-part archives using 7z...")
        try:
            # 'x' means extract with full paths
            # '-o' specifies output directory (no space after -o)
            # '-y' assumes Yes on all queries
            subprocess.run(
                ["7z", "x", str(main_zip), f"-o{target_path}", "-y"], check=True
            )
            print("✅ Extraction successful.")
        except subprocess.CalledProcessError as e:
            print(f"❌ 7z Extraction failed: {e}")
            print("💡 Ensure you ran: sudo apt install p7zip-full")
            return
    else:
        print("⏭️ Extraction folder already exists. Skipping to transcoding.")

    # --- PHASE 3: TRANSCODING (BATCH MODE) ---
    print("🎵 Transcoding FLAC -> WAV (16kHz, Mono)...")
    flac_files = list(target_path.rglob("*.flac"))
    total = len(flac_files)

    if total == 0:
        print("ℹ️ No FLAC files found to transcode.")
    else:
        for i, flac_path in enumerate(flac_files):
            wav_path = flac_path.with_suffix(".wav")

            if wav_path.exists():
                if flac_path.exists():
                    os.remove(flac_path)
                continue

            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(flac_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                str(wav_path),
            ]

            try:
                subprocess.run(cmd, check=True)
                os.remove(flac_path)  # Delete source to save space
                if (i + 1) % 1000 == 0:
                    print(f"  Progress: {i+1}/{total} files converted...")
            except:
                continue

    # --- PHASE 4: CLEANUP ---
    print("🧹 Cleaning up temp download folder...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print(f"\n🏁 ALL DONE! Data is ready in: {target_dir}")


if __name__ == "__main__":
    download_and_convert_to_wav()
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="cvssp/WavCaps",
        repo_type="dataset",
        local_dir="./original_data",
        allow_patterns=["json_files/AudioSet_SL/*.json"],
    )
