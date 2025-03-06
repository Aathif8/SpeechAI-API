# Importing Libraries
import os
import uvicorn
import openai
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Initializing FastAPI app
app = FastAPI()

# Ensure directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAI API Key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configuration
# MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = hf_hub_download(repo_id="Aathif/mistral-7b-instruct-v0.1.Q4_K_M.gguf", filename="mistral-7b-instruct-v0.1.Q4_K_M.gguf")
CHROMA_DB_PATH = "./chroma_db"


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

# Speech-to-Text Function
def Transcribe(audio_path):
    with open(audio_path, "rb") as audio_file:
        response = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return response.text

# Text-To-Speech Function
def generate_speech(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    output_audio_path = os.path.join(OUTPUT_DIR, "output.mp3")

    with open(output_audio_path, "wb") as audio_file:
        audio_file.write(response.content)
    return output_audio_path

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
    output_audio = generate_speech(response_text)

    return {
        "transcription": transcribed_text,
        "response": output_audio,
        "audio_file": f"/download_audio/"
    }

# @app.get("/download_audio/")
# async def download_audio():
#     output_audio_path = os.path.join(OUTPUT_DIR, "output.wav")

#     if not os.path.exists(output_audio_path):
#         raise HTTPException(status_code=404, detail="Audio file not found.")

#     return FileResponse(output_audio_path, media_type="audio/wav", filename="output.wav")

# Run FastAPI server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)