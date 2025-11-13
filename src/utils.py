import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import srt
import os
import tempfile
import warnings
import logging

def suppress_warnings_and_logs():
    # Silence Python warnings
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.filterwarnings("ignore")
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="pkg_resources is deprecated as an API.*"
    )

    # HF transformers verbosity
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    try:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # Silence logging globally
    logging.getLogger().setLevel(logging.ERROR)
    logging.getLogger("root").setLevel(logging.ERROR)
    logging.getLogger("chatterbox").setLevel(logging.ERROR)
    logging.getLogger("chatterbox.models.t3.inference.alignment_stream_analyzer").setLevel(logging.ERROR)

def to_mono_float32(y):
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.astype(np.float32, copy=False)

def resample_if_needed(y, original_sr, target_sr):
    if original_sr == target_sr:
        return to_mono_float32(y)
    return to_mono_float32(librosa.resample(y, orig_sr=original_sr, target_sr=target_sr))

def extract_audio_snippet(video, out_wav, start_sec, duration_sec, sr=24_000):
    (
        ffmpeg.input(video, ss=start_sec, t=duration_sec)
        .output(out_wav, vn=None, ac=1, ar=sr, acodec="pcm_s16le", y=None)
        .run(quiet=True)
    )
    return out_wav

def replace_audio_in_video(input_video, input_audio, output_video):
    video = ffmpeg.input(input_video)
    audio = ffmpeg.input(input_audio)
    (
        ffmpeg
        .output(video.video, audio.audio, output_video, vcodec="copy", acodec="aac", shortest=None)
        .overwrite_output()
        .run(quiet=True)
    )

def get_video_duration_seconds(path):
    try:
        info = ffmpeg.probe(path)
        vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams and "duration" in vstreams[0]:
            return float(vstreams[0]["duration"])
        if "format" in info and "duration" in info["format"]:
            return float(info["format"]["duration"])
    except Exception:
        pass
    return None

def extract_audio_from_subtitles(video_path, subtitles_path, sr=24000):
    """
    Extract audio snippet from video based on subtitle timings.
    If video is longer than 10 seconds, extracts 10 seconds from the longest subtitle.
    Otherwise extracts 5 seconds (or available duration if less).
    """
    video_duration = get_video_duration_seconds(video_path)
    if video_duration is None:
        raise ValueError("Could not determine video duration.")
    
    with open(subtitles_path, encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    
    if not subs:
        raise ValueError("No subtitles found in file.")

    longest_sub = None
    longest_duration = 0.0
    for sub in subs:
        start_time = sub.start.total_seconds()
        end_time = sub.end.total_seconds()
        sub_duration = end_time - start_time
        if sub_duration > longest_duration:
            longest_duration = sub_duration
            longest_sub = sub
    
    if longest_sub is None:
        raise ValueError("Could not find valid subtitle segments.")
    
    longest_sub_start = longest_sub.start.total_seconds()
    
    if video_duration > 10.0:
        duration = 10.0
    else:
        duration = min(5.0, longest_duration)
    
    # Ensure we don't exceed the longest subtitle duration or video duration
    remaining_in_subtitle = longest_sub.end.total_seconds() - longest_sub_start
    remaining_in_video = video_duration - longest_sub_start
    duration = min(duration, remaining_in_subtitle, remaining_in_video)
    if duration <= 0:
        raise ValueError("No audio available to extract based on subtitle timings.")

    output_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
    
    extract_audio_snippet(video_path, output_wav, longest_sub_start, duration, sr=sr)
    
    return output_wav

def loudnorm_wav(in_wav: str, out_sr: int = 24000, mono: bool = True):
    (
        ffmpeg
            .input(in_wav)
            .filter('loudnorm')
            .output(in_wav.replace(".wav", "_norm.wav"))
            .overwrite_output()
            .run(quiet=True)
    )
    os.replace(in_wav.replace(".wav", "_norm.wav"), in_wav)
    return in_wav
