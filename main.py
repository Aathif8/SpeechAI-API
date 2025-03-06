# Importing Libraries
import os
import soundfile as sf
import uvicorn
import librosa
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from transformers import pipeline
from TTS.api import TTS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
import time

# Initializing FastAPI app
app = FastAPI()

# Ensure directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = hf_hub_download(repo_id="Aathif/mistral-7b-instruct-v0.1.Q4_K_M.gguf", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")
CHROMA_DB_PATH = "./chroma_db"

# Load STT model
# processor = AutoProcessor.from_pretrained("distil-whisper/distil-small.en")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("distil-whisper/distil-small.en")
stt_model = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")

# Load TTS model
tts_model = TTS(model_name="tts_models/en/ljspeech/glow-tts", progress_bar=False)

# Load Mistral model
llm = LlamaCpp(model_path=MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count(), f16_kv=True, verbose=False)

# Global retriever
retriever = None

# Data Upload API
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    global retriever

    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(pdf_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process The File
    pdf_loader = PyPDFLoader(pdf_path)
    docs = pdf_loader.load()

    # Split into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Store in ChromaDB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory=CHROMA_DB_PATH)

    # Create retriever
    retriever = vectorstore.as_retriever()

    return {"message": "File Uploaded and Processed Successfully!"}

# Audio Processing API
def Transcribe(audio_path):
    audio, sr = sf.read(audio_path)

    # Resampling the audio
    target_sr = 16000
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Ensure audio is mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)  # Convert stereo to mono

    # Split into 30-second chunks
    chunk_size = sr * 30
    chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]

    full_transcription = []
    for chunk in chunks:
        # Convert chunk to a NumPy array and ensure float32 type
        chunk = np.array(chunk, dtype=np.float32)

        # Decode and store the transcription
        transcription = stt_model({"array": chunk, "sampling_rate": sr})["text"]
        full_transcription.append(transcription)

    final_transcription = " ".join(full_transcription)
    return final_transcription

@app.post("/process_audio/")
async def process_audio(file: UploadFile = File(...)):
    global retriever

    # Save the uploaded file
    audio_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(audio_path, "wb") as buffer:
        buffer.write(await file.read())

    # Transcription of Audio
    transcribed_text = Transcribe(audio_path)

    # Query LLM (RAG Model)
    if retriever is None:
        return {"error": "No document uploaded for response"}

    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = rag_chain.invoke({"query": transcribed_text})

    # Extract response text
    response_text = response["result"] if isinstance(response, dict) and "result" in response else str(response)

    # Text to Speech Response
    output_audio_path = os.path.join(OUTPUT_DIR, "output.wav")

    try:
        tts_model.tts_to_file(text=response_text, file_path=output_audio_path)
        print(f"Audio file generated at: {output_audio_path}")

        # Ensure file exists before returning response
        time.sleep(2)  # Wait for file generation
        if not os.path.exists(output_audio_path):
            raise HTTPException(status_code=500, detail="TTS model failed to generate audio file.")

    except Exception as e:
        print(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {e}")

    return {
        "transcription": transcribed_text,
        "response": response_text,
        "audio_file": f"/download_audio/"
    }

@app.get("/download_audio/")
async def download_audio():
    output_audio_path = os.path.join(OUTPUT_DIR, "output.wav")

    if not os.path.exists(output_audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
