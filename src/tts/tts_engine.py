import numpy as np
import soundfile as sf
import torchaudio as ta
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from typing import Tuple


class ChatterboxTTSEngine:
    def __init__(self, device: str | None = None):
        if ChatterboxMultilingualTTS is None:
            raise RuntimeError("Chatterbox-TTS is not installed.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

    def synthesize_speech(self, text: str, speaker_ref: str, dest_language: str, cfg_weight: float, tmp_file: str) -> Tuple[np.ndarray, int]:
        # Keep cfg_weight at 0.0 when reference clip language != target to avoid accent leakage (Reference: https://huggingface.co/ResembleAI/chatterbox)
        wav = self.model.generate(text, audio_prompt_path=speaker_ref, language_id=dest_language, cfg_weight=cfg_weight)
        ta.save(tmp_file, wav, self.model.sr)
        audio, sr = sf.read(tmp_file)
        return audio, sr