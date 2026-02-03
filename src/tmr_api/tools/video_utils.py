import os
import urllib.request
import subprocess

def download_and_trim_video(babel_id, start_time, end_time, output_path):
    """
    Downloads the video from BABEL S3 and trims it using ffmpeg.
    """
    url = f"https://babel-renders.s3.eu-central-1.amazonaws.com/{babel_id}.mp4"
    # Use a temp filename that includes the PID or random string to avoid collisions if running in parallel,
    # but for now a simple name is fine for CLI usage.
    temp_video = f"temp_{babel_id}.mp4"
    
    try:
        print(f"  Downloading video from {url}...")
        urllib.request.urlretrieve(url, temp_video)
        
        # ffmpeg command to trim
        # -y to overwrite output
        cmd = [
            "ffmpeg",
            "-y",
            "-i", temp_video,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "libx264",
            "-c:a", "aac", 
            "-strict", "experimental",
            output_path
        ]
        
        print(f"  Trimming video {start_time}s to {end_time}s...")
        # Use subprocess.run to execute ffmpeg
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f"  Saved video to {output_path}")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"  [Error] Failed to download video: {e}")
    except subprocess.CalledProcessError as e:
        print(f"  [Error] ffmpeg failed: {e.stderr.decode()}")
    except Exception as e:
        print(f"  [Error] An unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_video):
            os.remove(temp_video)
    return False
