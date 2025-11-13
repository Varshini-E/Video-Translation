from dataclasses import dataclass
from typing import Optional, Literal
import os
import numpy as np
import soundfile as sf
import tempfile
import librosa
import srt
from tqdm import tqdm
from src.utils import resample_if_needed
from src.tts.tts_engine import ChatterboxTTSEngine


@dataclass
class CloneConfiguration:
    language_id: str = "de"
    target_sr: int = 24_000
    cfg_weight: float = 0.0
    alignment: Literal["global", "local"] = "global"
    max_stretch: float = 1.2
    overflow_tolerance: float = 0.1

class VoiceCloner:
    """
    Class for cloning a speaker from a short reference clip into the target language with the given text and synthesizing a new speech with Chatterbox.
    """
    def __init__(self, cfg: Optional[CloneConfiguration] = None):
        self.cfg = cfg or CloneConfiguration()
        self.temp_dir = tempfile.TemporaryDirectory(prefix="voice_cloner_")
        self.tts = ChatterboxTTSEngine()
    
    def _get_segment_length(self, audio: np.ndarray) -> float:
        return len(audio) / self.cfg.target_sr

    def _global_align_audio(self, y: np.ndarray, sr: int, target_dur: float, r_min: float = 0.80, r_max: float = 1.25) -> np.ndarray:
        if y.size == 0 or not target_dur or target_dur <= 0:
            return y
        epsilon = 1e-6

        cur_dur = len(y) / sr
        if abs(cur_dur - target_dur) < 1e-3:
            return y
        
        ratio = cur_dur / target_dur
        # >1 means audio is longer than target, need to speed up
        if ratio > 1 + epsilon:
            while ratio > r_max + 1e-3:
                y = librosa.effects.time_stretch(y, rate=r_max)
                cur_dur = len(y) / sr
                ratio = cur_dur / target_dur
            y = librosa.effects.time_stretch(y, rate=max(ratio, epsilon))
        # <1 means audio is shorter than target, need to slow down
        elif ratio < 1 - epsilon:
            while ratio < r_min - 1e-3:
                y = librosa.effects.time_stretch(y, rate=r_min)
                cur_dur = len(y) / sr
                ratio = cur_dur / target_dur
            y = librosa.effects.time_stretch(y, rate=max(ratio, epsilon))
        
        target_samples = int(round(target_dur * sr))
        if len(y) > target_samples:
            y = y[:target_samples]
        elif len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)), mode="constant")
        return y
    
    def _local_align_audio(self, seg_audio: np.ndarray, target_dur: float) -> np.ndarray:
        factor = target_dur / self._get_segment_length(seg_audio)
        rate = 1 / factor
        if (1 / self.cfg.max_stretch) < rate < self.cfg.max_stretch:
            seg_audio = librosa.effects.time_stretch(seg_audio, rate=rate)

            # After stretch: enforce alignment vs subtitle window
            if self._get_segment_length(seg_audio) < target_dur:
                # too short: pad with silence to fill slot
                pad = int((target_dur - self._get_segment_length(seg_audio)) * self.cfg.target_sr)
                seg_audio = np.concatenate([seg_audio, np.zeros(pad, dtype=np.float32)])
            else:
                seg_audio = seg_audio[:int(target_dur * self.cfg.target_sr)]
            
        return seg_audio.astype(np.float32, copy=False)

    
    def _clone_with_alignment(self, subtitles_srt: str, speaker_ref: str, video_duration: float) -> str:
        with open(subtitles_srt, encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))

        # Generate TTS for each subtitle segment and stitch
        full_audio = np.array([], dtype=np.float32)
        current_time = 0.0

        for sub in tqdm(subs, desc="Generating voice cloned segments", unit="segment"):
            text = sub.content.strip() 
            if not text:
                continue
            start_time = sub.start.total_seconds()
            end_time = sub.end.total_seconds()
            duration = end_time - start_time
            target_dur = max(0.1, duration)

            # Insert silence for gaps
            if start_time > current_time:
                gap = start_time - current_time
                if gap > 0:
                    silence = np.zeros(int(round(gap * self.cfg.target_sr)), dtype=np.float32)
                    full_audio = np.concatenate([full_audio, silence])
                    current_time += gap

            # Generate with Chatterbox 
            tmp_seg = os.path.join(self.temp_dir.name, f"seg_{sub.index}.wav")
            seg_audio, seg_sr = wav = self.tts.synthesize_speech(text, speaker_ref, self.cfg.language_id, self.cfg.cfg_weight, tmp_seg)

            if seg_audio.ndim > 1:
                seg_audio = seg_audio.mean(axis=1)
            seg_audio = resample_if_needed(seg_audio.astype(np.float32), seg_sr, self.cfg.target_sr)

            if self.cfg.alignment == "local":
                seg_audio = self._local_align_audio(seg_audio, target_dur)

            current_time += self._get_segment_length(seg_audio)
            full_audio = np.concatenate([full_audio, seg_audio])

        if self.cfg.alignment == "global":
            if video_duration and full_audio.size > 0:
                full_audio = self._global_align_audio(full_audio, self.cfg.target_sr, video_duration)
            
        output_audio_wav = os.path.join(self.temp_dir.name, "cloned_output.wav")
        sf.write(output_audio_wav, full_audio, self.cfg.target_sr)
        return output_audio_wav
    
    def clone_speech(self, subtitles: str, speaker_ref: str, video_duration: float) -> str:
        return self._clone_with_alignment(subtitles, speaker_ref, video_duration)
