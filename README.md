# **Video-Translate**
A tool that translates video from English to German while keeping the speaker’s voice and the video closely in sync. 

## **Setup**
Setting up the environment and running inference is simple and the solution works on both CPU and GPU devices. 
For Text-to-Speech Voice Cloning, the [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) model is utilized which requires a small fix to be carried out on CPU-only devices [a known issue yet to be fixed on Chatterbox's end]. 

### **1. Create a Conda Environment**

First, create a new conda environment named video-translate by running the following command:

```bash
conda create -n face python=3.10
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
3. (Optional Lip-sync)

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
Extract subtitle timing from the source `.srt`, translate each segment to the target language, and prepare structured German subtitle text for converting to speech.

### 2. Voice Cloning & Audio Generation
Extract a short speaker-reference clip, clone the speaker’s voice, and synthesize the translated lines from the subtitles. Align segments and apply global tempo correction with selective local tempo correction so the final audio duration matches the video.

### 3. Audio–Video Reconstruction
Replace the original audio track with the synthesized one while keeping all video frames unchanged, producing a fully synchronized German-dubbed video.


# **Results**
Results from a sample run are available in the `output/` folder along with the input in the `data/` folder. 

### Original (with audio)
[🎥 Watch Original Video](data/Tanzania-2.mp4)

### Translated (with audio)
[🎥 Watch Translated Video](output/translated_video.mp4)



