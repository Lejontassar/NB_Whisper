
# NB-Whisper API (Serverless Ready for RunPod)

This FastAPI app supports multiple NB-Whisper models including standard and semantic versions. 

## Usage

POST /transcribe/

Form fields:
- `file`: audio file (WAV/MP3)
- `model`: one of `tiny`, `base`, `small`, `medium`, `large`, `tiny-semantic`, etc.

Response:
```json
{ "transcription": "Your transcribed text" }
```
