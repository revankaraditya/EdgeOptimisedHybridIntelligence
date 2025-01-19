from flask import Flask, request,jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from flask import render_template
import threading
import whisper
import sounddevice as sd
import numpy as np
import wave
import os

app = Flask(__name__)

SAMPLE_RATE = 16000
OUTPUT_FILE = "triggered_audio.wav"
RECORDING = False
AUDIO_DATA = []
RECOGNITION_THRESHOLD = 0.5  # Adjust based on environment

# Load Whisper model once (offline)
whisper_model = whisper.load_model("small")

VECTOR_STORAGE_PATH = "db"

folder_path = "db"

cached_llm = Ollama(model="git-gpt-v3")

embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST]You are a knowledgeable and helpful AI assistant. 
            Based on the below context [/INST] </s>
    [INST]  Context: {context}
            Answer the below question
            Question: {input}
            Provide only the answer
    [/INST]
"""
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/start', methods=['POST'])
def start_recording():
    """Start recording audio."""
    global RECORDING
    if RECORDING:
        return jsonify({"status": "Recording already in progress."}), 400

    RECORDING = True
    threading.Thread(target=record_audio).start()
    return jsonify({"status": "Recording started."})


@app.route('/stop', methods=['POST'])
def stop_recording():
    """Stop recording audio."""
    global RECORDING
    if not RECORDING:
        return jsonify({"status": "No recording in progress."}), 400

    RECORDING = False
    save_audio(OUTPUT_FILE, np.concatenate(AUDIO_DATA, axis=0))
    text = transcribe_and_execute(OUTPUT_FILE)
    return text[1:]


@app.route('/status', methods=['GET'])
def recording_status():
    """Get the current recording status."""
    return jsonify({"recording": RECORDING})


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")
    print(f"query: {query}")

    if is_college_related(query):
        return rag_query(query)
    return normal_query(query)


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response


@app.route('/admin/')
def about():
    return render_template('admin.html')


@app.route('/delete-vector-storage', methods=['POST'])
def delete_vector_storage():
    try:
        # Check if the folder exists
        if os.path.exists(VECTOR_STORAGE_PATH):
            # Remove all files and subdirectories within the folder
            for root, dirs, files in os.walk(VECTOR_STORAGE_PATH, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))

            # Remove the folder itself
            os.rmdir(VECTOR_STORAGE_PATH)

            return jsonify({"status": "success", "message": "Vector storage deleted successfully."}), 200
        else:
            return jsonify({"status": "error", "message": "Vector storage does not exist."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def normal_query(query):
    print(f"Normal")
    response = cached_llm.invoke(query)
    print(response)
    response_answer = {"answer": response}
    return response_answer["answer"]


def rag_query(query):
    print(f"RAG")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.2,
        },
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer["answer"]


def is_college_related(query):
    college_keywords = {
        "university", "college", "campus", "professor", "student", "teacher",
        "faculty", "class", "classes", "exam", "exams", "test", "tests",
        "quiz", "quizzes", "course", "courses", "assignment", "assignments",
        "project", "projects", "syllabus", "curriculum", "lecture", "lectures",
        "degree", "bachelor", "master", "phd", "dissertation", "thesis",
        "academic", "academics", "education", "semester", "credit", "credits",
        "internship", "internships", "placement", "library", "hostel",
        "department", "graduation", "postgraduate", "undergraduate",
        "admissions", "scholarship", "scholarships", "tuition", "fee",
        "fees", "research", "paper", "papers", "conference", "mentor",
        "tutor", "lab", "labs", "labwork", "practical", "practicals",
        "dorm", "hostel", "study", "studies", "classroom", "quiz", "kls git", "admission",
        "intake", "kls", "#", "placement"
    }
    if any(keyword in query.lower() for keyword in college_keywords):
        return True
    return False


# voice Functions
def save_audio(filename, audio_data):
    """Save audio data to a WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())


def record_audio():
    """Record audio until stopped."""
    global RECORDING, AUDIO_DATA

    def audio_callback(indata, frames, time, status):
        if RECORDING:
            AUDIO_DATA.append(indata.copy())

    AUDIO_DATA = []
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
        while RECORDING:
            sd.sleep(100)  # Sleep to avoid busy-waiting


def transcribe_and_execute(filename):
    """Transcribe the recorded audio and execute actions based on transcription."""
    transcription = whisper_model.transcribe(filename)
    text = transcription["text"]
    print(f"Transcription: {text}")
    return text
    # pyautogui.typewrite(text[1:])
    # pyautogui.typewrite(['enter'])


def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
