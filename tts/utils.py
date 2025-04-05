# tts/utils.py
import os
import time
from typing import Optional

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def generate_filename(prefix: str = "tts") -> str:
    return f"{prefix}_{int(time.time()*1000)}.wav"

def validate_audio_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")
    if not path.lower().endswith('.wav'):
        raise ValueError("仅支持WAV格式音频")