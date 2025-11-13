import os
import srt
import shutil
import soundfile as sf
from src.tts.voice_cloning import CloneConfiguration, VoiceCloner
from src.translate_text.translator import SimpleTranslator
from src.utils import replace_audio_in_video

class VideoTranslator:
    def __init__(self, input_video: str, subtitles: str, output_path: str, output_video_name: str, tts_config: CloneConfiguration, speaker_ref: str, video_duration: float):
        self.input_video = input_video
        self.subtitles = subtitles
        self.voice_cloner = VoiceCloner(tts_config)
        self.translator = SimpleTranslator()

        os.makedirs(output_path, exist_ok=True)
        self.output_video = os.path.join(output_path, f"{output_video_name}.mp4")
        self.output_path = output_path
        self.speaker_ref = speaker_ref
        self.video_duration = video_duration

    def translate_video(self) -> str:
        translated_subtitles = self.translator.translate_subs(self.subtitles)
        print("Subtitles translated to German\n")
        translated_audio = self.voice_cloner.clone_speech(translated_subtitles, self.speaker_ref, self.video_duration)
        print("Text to speech in cloned voice generated\n")
        replace_audio_in_video(self.input_video, translated_audio, self.output_video)
        print(f"Video translated and saved to {self.output_video}")
        return self.output_video