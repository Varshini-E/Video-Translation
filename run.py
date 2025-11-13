
import argparse
from src.pipeline.video_translation import VideoTranslator
from src.tts.voice_cloning import CloneConfiguration
from src.utils import get_video_duration_seconds, extract_audio_from_subtitles

DEFAULT_CLONE_CONFIG = CloneConfiguration(
    language_id="de",
    target_sr=24_000,
    cfg_weight=0.0,
    alignment="global",
    max_stretch=1.2,
    overflow_tolerance=0.1
)

def main():
    parser = argparse.ArgumentParser("Translate a video from English to German")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--subtitles", type=str, required=True, help="Path to the subtitles file.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to the output directory.")
    parser.add_argument("--output_video_name", type=str, default="translated-video", help="Name of the output video file.")
    parser.add_argument("--cfg_weight", type=float, default=0.0, help="CFG weight for the audio cloning.")
    parser.add_argument("--alignment", type=str, default="global", choices=["global", "local"], help="Alignment method for the audio.")
    parser.add_argument("--sr", type=int, default=24000, help="Sample rate for the audio.")
    parser.add_argument("--max_stretch", type=float, default=1.2, help="Maximum stretch for the audio.")
    parser.add_argument("--overflow_tolerance", type=float, default=0.1, help="Overflow tolerance for the audio.")
    args = parser.parse_args()

    video_duration = get_video_duration_seconds(args.input_video)
    if video_duration is None:
        raise ValueError("Could not determine video duration.")
    
    speaker_ref = extract_audio_from_subtitles(args.input_video, args.subtitles, args.sr)
    TTSConfig = DEFAULT_CLONE_CONFIG
    TTSConfig.cfg_weight = args.cfg_weight
    TTSConfig.alignment = args.alignment
    TTSConfig.target_sr = args.sr
    TTSConfig.max_stretch = args.max_stretch
    TTSConfig.overflow_tolerance = args.overflow_tolerance

    video_translator = VideoTranslator(args.input_video, args.subtitles, args.output_path, args.output_video_name, TTSConfig, speaker_ref, video_duration)
    video_translator.translate_video()

if __name__ == "__main__":
    main()
