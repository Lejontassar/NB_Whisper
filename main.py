
from fastapi import FastAPI, UploadFile, File, Form
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import io

app = FastAPI()

MODEL_MAP = {
    "tiny": "NbAiLab/nb-whisper-tiny",
    "base": "NbAiLab/nb-whisper-base",
    "small": "NbAiLab/nb-whisper-small",
    "medium": "NbAiLab/nb-whisper-medium",
    "large": "NbAiLabBeta/nb-whisper-large",
    "tiny-semantic": "NbAiLab/nb-whisper-tiny-semantics",
    "base-semantic": "NbAiLab/nb-whisper-base-semantics",
    "small-semantic": "NbAiLab/nb-whisper-small-semantics",
    "medium-semantic": "NbAiLab/nb-whisper-medium-semantics",
    "large-semantic": "NbAiLabBeta/nb-whisper-large-semantics"
}

loaded_models = {}

def get_model_and_processor(model_key):
    if model_key not in loaded_models:
        processor = WhisperProcessor.from_pretrained(MODEL_MAP[model_key])
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_MAP[model_key])
        loaded_models[model_key] = (processor, model)
    return loaded_models[model_key]

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...), model: str = Form("base")):
    if model not in MODEL_MAP:
        return {"error": f"Model '{model}' is not supported."}

    waveform, sample_rate = torchaudio.load(io.BytesIO(await file.read()))
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

    processor, model_instance = get_model_and_processor(model)
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    generated_ids = model_instance.generate(inputs["input_features"])
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return {"transcription": transcription}
