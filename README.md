# **Video-Translate**
A tool that translates video from English to German while keeping the speaker’s voice and the video closely in sync. 

## **Setup**
Setting up the environment and running inference is simple and the solution works on both CPU and GPU devices. 
For Text-to-Speech Voice Cloning, the [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) model is utilized which requires a small fix to be carried out on CPU-only devices [a known issue yet to be fixed on Chatterbox's end]. 

### **1. Create a Conda Environment**

First, create a new conda environment named video-translate by running the following command:

```bash
conda create -n video-translate python=3.10
conda activate video-translate

```
### **2. Install Dependencies**
```
pip install -r requirements.txt
```
###  **3. [CPU-only devices] Fix environment specific setup for the Chatterbox model**
By default, Chatterbox expects a CUDA GPU.  
If you're running on CPU, you need to modify two model loading lines in the installed environment package:

File: `<your_conda_env>/lib/python3.10/site-packages/chatterbox/mtl_tts.py`  

Change these lines:
```python
ve = VoiceEncoder()
ve.load_state_dict(
    torch.load(ckpt_dir / "ve.pt", weights_only=True)
)
...
s3gen = S3Gen()
s3gen.load_state_dict(
    torch.load(ckpt_dir / "s3gen.pt", weights_only=True)
)
```
to
```python
ve = VoiceEncoder()
ve.load_state_dict(
    torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=device)
)
...
s3gen = S3Gen()
s3gen.load_state_dict(
    torch.load(ckpt_dir / "s3gen.pt", weights_only=True, map_location=device)
)
```
where `map_location=device` is additionally passed to the `torch.load` function. 

## **Inference on a Custom Video + Subtitles**

1. **Prepare the Data:**  
   - Add any English language video (`video.mp4`) and its corresponding subtitles (`subtitles.srt`) to the `data/` folder.  

2. **Run the Pipeline:**  
   Execute the following command:  
   ```bash
    python run.py --input_video "data/video.mp4" \
                  --subtitles "data/subtitles.srt" 
   ```

List of CLI arguments for `run.py`
### Command-Line Arguments

| Argument | Type | Default | Description |
|---------|------|----------|-------------|
| `--input_video` | `str` | **required** | Path to the input video file. |
| `--subtitles` | `str` | **required** | Path to the subtitle (`.srt`) file. |
| `--output_path` | `str` | `"output"` | Directory where outputs will be saved. |
| `--output_video_name` | `str` | `"translated-video"` | Name of the final exported video file. |
| `--cfg_weight` | `float` | `0.0` | CFG weight used by Chatterbox for voice cloning (0 recommended). |
| `--alignment` | `str` | `"global"` | Audio alignment mode: `"global"` or `"local"`. |
| `--sr` | `int` | `24000` | Sample rate used for synthesis. |
| `--max_stretch` | `float` | `1.2` | Maximum stretch factor for local alignment. |
| `--overflow_tolerance` | `float` | `0.1` | Allowed overflow before segment trimming. |

3. **Optional: Lip-Sync**

Lip-sync is *not included* in this MVP since all lip-sync models require GPUs. 
The core deliverable focuses on audio translation, voice cloning, and synchronization with the original video timing.

To experiment with basic lip-sync for a short clipping, here's a Colab notebook to be executed with a GPU runtime: 
Lip-Sync <a target="_blank" href="https://colab.research.google.com/drive/13yX2A66Pwz-qldFroiqYYFY8lD54FjlJ?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a> 

For a 10-second clip, it takes close to 6-7 minutes to generate the corresponding lip-synced video. Though some portions align well, multiple artifacts and out-of-sync portions are present. 

For a production-level pipeline, models like LatentSync run on a GPU compatible system would suffice to give a lip-synced video.




## Pipeline Overview

This project takes a source video in the English language along with the transcribed subtitles and produces a translated, voice-cloned version in the German language.  
The main steps include subtitle translation, text-to-speech using voice cloning and overlaying the translated audio back to the original source video.

### Inputs
- A video file with original audio  
- The original transcription of the audio  

### Output Artifacts

| File Type | Description |
|------|--------------|
| .srt | Translated subtitles |
| .wav | Generated voice-cloned audio |
| .mp4 | Final video with synchronized translated speech |

### 1. Subtitle Processing & Translation
Extract subtitle timing from the source `.srt`, translate each segment to the target language using a dedicated [English to German Machine Translation model](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) , and prepare structured German subtitle text for converting to speech.

### 2. Voice Cloning & Audio Generation
Extract a short speaker-reference clip, clone the speaker’s voice, and synthesize the translated lines from the subtitles using a [Multilingual TTS model](https://huggingface.co/ResembleAI/chatterbox). Align segments and apply global tempo correction with selective local tempo correction so the final audio duration matches the video.

### 3. Audio–Video Reconstruction
Replace the original audio track with the synthesized one while keeping all video frames unchanged, producing a fully synchronized German-dubbed video.

### 4. (Optional with separate Colab setup) Lip-Sync Enhancement  
Uses [Wav2Lip-GFPGAN](https://github.com/ajay-sainy/Wav2Lip-GFPGAN) based on [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) with [GFPGAN](https://github.com/TencentARC/GFPGAN) for high-fidelity lip synchronization.  
The original video without audio and the synthesized German audio is fed into the model, which aligns the speaker’s lip movements frame-by-frame with the new speech, while GFPGAN enhances facial realism and visual consistency producing a naturally dubbed, photorealistic final video.


## **Results**
Results from a sample run are available in the `output/` folder along with the input in the `data/` folder. 

### Original (with audio)
[🎥 Watch Original Video](data/Tanzania-2.mp4)

### Translated (with audio)
[🎥 Watch Translated Video](output/translated_video.mp4)

## **Assumptions & Limitations of the Pipeline**

### Assumptions
- The input video contains one primary speaker in English whose voice can be reliably cloned.
- The provided subtitle file (.srt) has accurate timestamps that correctly reflect spoken dialogue.
- The translation is sentence-aligned and does not drastically change speaking duration.
- The extracted speaker reference (5–10 seconds) is clean, with minimal background noise.
- The video has no rapid multi-speaker cuts, heavy background noise, or complex transitions.

### Limitations
- Chatterbox TTS is slow on CPU, especially for long videos.
- Global audio alignment speeds up or slows down the entire audio track to match the original video duration; this preserves fluency but cannot fix per-sentence pacing differences, and long silences in the video may desynchronize.
- Local audio alignment adjusts each subtitle segment individually; this can cause overlaps, abrupt cutoffs, or synthetic timing artifacts.
- Extreme mismatch between TTS duration and original timing may cause audio artifacts even with global alignment.
- Background music, noise, or SFX cannot be isolated as the pipeline fully replaces the audio track.
- No lip-sync correction is included in the main pipeline and mouth movements will not match the translated audio.

## **Design Choices**

**[Coqui TTS (XTTS v2)](https://huggingface.co/coqui/XTTS-v2)**

**Pros:** Fast inference, lightweight, easier to run on CPU, good overall voice quality with more natural German accent.

**Cons:** Speech duration often drifts from subtitle timing for German language, leading to longer total audio and requiring aggressive time-stretching, which can degrade naturalness.

**[Chatterbox TTS](https://huggingface.co/ResembleAI/chatterbox)**

**Pros:** Generates speech that is more tightly aligned to the original timing, reducing the need for trimming or stretch-correction. Produces more stable pacing for dubbing. 

**Cons:** Significantly slower on CPU-only environments and requires a minor internal fix for CPU deserialization. In multilingual settings, accent leakage tends to happen from one language to another. 

Final Choice:
Chatterbox TTS is used as the default due to its more consistent temporal alignment, which yields better synchronization in the final dubbed video, despite being slower and slightly poorer in accent quality.


