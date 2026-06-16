from sentence_transformers import SentenceTransformer
import numpy as np# Small + fast embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # free, tiny, perfect

def chunk_text(text, chunk_size=800):
    """
    Split PDF text into clean chunks of ~800 characters each.
    """
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) > chunk_size:
            chunks.append(" ".join(chunk))
            chunk = []

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def create_embeddings(chunks):
    """
    Convert all chunks to vector embeddings.
    """
    vectors = embedder.encode(chunks, convert_to_numpy=True)
    return vectors


def retrieve_top_k(query, chunks, vectors, k=3):
    """
    Retrieve the top-K relevant chunks for the query.
    """
    q_vec = embedder.encode([query], convert_to_numpy=True)[0]
    scores = np.dot(vectors, q_vec)
    top_k_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_k_idx]



from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfMerger, PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
import os
from PIL import Image
from docx import Document
from reportlab.lib.pagesizes import letter
PDF_CHAT_SESSIONS = {}

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------------------------------
# HOME
# ------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

from flask import render_template

import os
import math
from flask import Flask, request, render_template, redirect, url_for, flash
import PyPDF2
from dotenv import load_dotenv

# # Try to use official Groq SDK if installed, otherwise fallback to requests
try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    import requests
    GROQ_SDK_AVAILABLE = False

load_dotenv()  # optional, for local .env

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Set GROQ_API_KEY environment variable before running.")
# from google import genai

# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise RuntimeError("Set GEMINI_API_KEY environment variable before running.")

# # gemini_client = genai.Client(api_key=GEMINI_API_KEY)
# import requests
# DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
# DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"


app.secret_key = os.environ.get("FLASK_SECRET", "change-me")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Groq client if SDK available
if GROQ_SDK_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None
    GROQ_REST_URL = "https://api.groq.com/v1"  # keep generic; endpoints below use model path

# ============ Prompt templates per format ============
PROMPTS = {
    "simple": {
        "system": "You are a helpful assistant. Produce a clear, concise paragraph summary focused on the main ideas. Keep language natural and easy to read.",
        "instruction": "Summarize the following content in one concise paragraph highlighting the main points and purpose."
    },
    "bullets": {
        "system": "You are a helpful assistant. Produce a short, informative bullet-point summary that a reader can scan quickly.",
        "instruction": "Summarize the following content as 5-10 succinct bullet points with key facts, takeaways, and any important numbers or actions."
    },
    "abstract": {
        "system": "You are an academic assistant. Produce a research-style abstract with clarity and formal tone.",
        "instruction": "Create a 150-250 word abstract suitable for an academic paper: include background, methods (if present), core findings, and a concise conclusion."
    },
    "legal": {
        "system": "You are a legal analyst. Produce a clear, structured legal brief with neutral, formal language.",
        "instruction": "Summarize the following text as a legal brief covering: facts, key legal issues, likely interpretations, and recommended next steps or conclusions."
    },
    "teaching": {
        "system": "You are an instructor and pedagogue. Explain the content step-by-step as if teaching a beginner, using simple language and examples.",
        "instruction": "Summarize and teach the following material. Include: a short overview, 3–5 key concepts, simple examples, a small Q&A (2 sample questions + answers), and key takeaways."
    },
    "short": {
        "system": "You are a concise assistant. Give an ultra-short summary.",
        "instruction": "Provide a single-sentence summary of the content."
    },
    "long": {
        "system": "You are a thorough summarizer and analyst. Produce a detailed structured summary that a reader can rely on without seeing the original.",
        "instruction": "Provide a detailed summary with headings, subpoints, and examples. Include: Overview, Key Points, Detailed Explanation (with examples), and Actionable Recommendations. Aim for 300–700 words."
    },

    # Strict JSON helpers (for structured outputs like chapters / QA)
    "strict_chapters": {
        "system": "You are a strict extractor. Output must be valid JSON only — no commentary, no markdown.",
        "instruction": (
            "Extract chapters as JSON array objects with fields: title, content. "
            "Output ONLY JSON structured exactly as: {\"chapters\":[{\"title\":\"...\",\"content\":\"...\"}, ... ]}"
        )
    },
    "strict_questions": {
        "system": "You are an exam generator. Output STRICT JSON only.",
        "instruction": (
            "From the provided chapter text produce JSON with: chapter, easy (5 items), medium (5), hard (5), mcq (10 items with options, answer, explanation). "
            "Output ONLY valid JSON exactly matching the specified schema."
        )
    }
}

# ============ Utilities ============
def extract_text_from_pdf(path):
    reader = PyPDF2.PdfReader(path)
    full_text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            full_text += t + "\n"
    return full_text

def chunk_text_chars(text, chunk_size=3000, overlap=300):
    """
    Chunk text by characters with optional overlap.
    chunk_size approx char length; overlap keeps continuity.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap ensures context continuity
    return chunks
# from google import genai

# 1. Initialize the client (New SDK style)
# os.environ["GOOGLE_API_KEY"] = "AIzaSyAuyQdWyyyxk"
# ... imports ...

# PASTE YOUR NEW KEY HERE AGAIN TO BE SURE
# my_secret_key = "AIzaSyAuyQdWyyyxkLg-r5SCSTciU1k" 

# print(f"DEBUG CHECK: The key I am using starts with: {my_secret_key[:10]}") # <--- ADD THIS

from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def call_groq_chat(messages, model="llama-3.1-8b-instant", max_new_tokens=1200, temperature=0.3):

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print("Groq Error:", e)
        return ""
    
    
@app.route("/ai-summarize", methods=["GET", "POST"])
def ai_summarize():
    if request.method == "POST":
        f = request.files.get("pdf")
        style = request.form.get("style", "simple")

        if not f:
            flash("Please upload a PDF file.", "error")
            return redirect(request.url)

        filename = f.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(file_path)

        # 1) Extract text
        full_text = extract_text_from_pdf(file_path)
        if not full_text.strip():
            return render_template("ai_summarize.html", summary="No extractable text.", style=style)

        # 2) Smaller chunk size (token reduction)
        chunks = chunk_text_chars(full_text, chunk_size=3000, overlap=300)

        chunk_summaries = []
        compressed_chunks = []   # token-optimized summaries for meta step

        # 3) Summarize + Compress each chunk (TWO-STEP optimization)
        for idx, chunk in enumerate(chunks, start=1):
            p = PROMPTS.get(style, PROMPTS["simple"])

            # --- Step A: Generate summary for each chunk ---
            messages = [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["instruction"] + "\n\n" + chunk}
            ]
            try:
                summary = call_groq_chat(messages, max_new_tokens=180)
            except Exception as e:
                summary = f"[error summarizing chunk {idx}: {str(e)}]"

            chunk_summaries.append(summary)

            # --- Step B: Compress summary for meta step (token saver) ---
            compress_msg = [
                {"role": "system", "content": "You compress summaries."},
                {"role": "user", "content": "Compress this summary into concise bullet points under 120 tokens:\n\n" + summary}
            ]
            try:
                compressed = call_groq_chat(compress_msg, max_new_tokens=120)
            except Exception as e:
                compressed = f"[error compressing chunk {idx}: {str(e)}]"

            compressed_chunks.append(compressed)

        # 4) Final meta-summary from COMPRESSED chunks (BIG TOKEN SAVING)
        merged = "\n\n".join(compressed_chunks)

        meta_prompt = PROMPTS.get(style, PROMPTS["simple"])
        meta_messages = [
            {"role": "system", "content": meta_prompt["system"]},
            {
                "role": "user",
                "content": (
                    "Using the compressed bullet points from each chunk, "
                    "produce a coherent summary in the requested style. "
                    "Combine, deduplicate, and ensure smooth flow.\n\n" + merged
                )
            }
        ]

        try:
            final_summary = call_groq_chat(meta_messages, max_new_tokens=250)
        except Exception as e:
            final_summary = "Error producing final summary: " + str(e)

        # 5) Render output
        return render_template(
            "ai_summarize.html",
            summary=final_summary,
            chunk_summaries=chunk_summaries,
            style=style,
            filename=filename
        )

    return render_template("ai_summarize.html", summary=None, style="simple")
import re
import json
import os
from PyPDF2 import PdfReader
from flask import request, render_template

@app.route("/ai-questions", methods=["GET", "POST"])
def ai_questions():
    if request.method == "POST":
        pdf = request.files.get("pdf")
        if not pdf:
            return render_template("ai_questions.html", error="Upload a PDF")

        filepath = os.path.join(UPLOAD_FOLDER, pdf.filename)
        pdf.save(filepath)

        # -----------------------------------
        # STEP 1: EXTRACT TEXT FROM PDF
        # -----------------------------------
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"

        if not text.strip():
            return render_template("ai_questions.html", error="No extractable text in PDF")

        # -----------------------------------
        # STEP 2: DETECT CHAPTERS (STRICT JSON)
        # -----------------------------------
        chapter_prompt = f"""
You extract chapters from textbooks.

STRICT OUTPUT RULES:
- Output ONLY valid JSON
- No markdown
- No explanations

FORMAT (EXACT):

{{
  "chapters": [
    {{
      "title": "Chapter name",
      "content": "Chapter text"
    }}
  ]
}}

Extract chapters from the text below:

{text[:6000]}
"""

        chapter_raw = call_groq_chat(
            [{"role": "user", "content": chapter_prompt}],
            max_new_tokens=900
        )

        chapter_clean = re.sub(r"```json|```", "", chapter_raw).strip()

        try:
            parsed = json.loads(chapter_clean)
            chapters = parsed.get("chapters", [])
        except:
            print("\n❌ BAD CHAPTER JSON:\n", chapter_raw)
            return render_template(
                "ai_questions.html",
                error="AI failed to detect chapters. Try a smaller PDF."
            )

        # -----------------------------------
        # STEP 3: GENERATE QUESTIONS
        # -----------------------------------
        final_output = []

        for ch in chapters:
            q_prompt = f"""
You generate exam questions.

STRICT OUTPUT RULES:
- Output ONLY valid JSON
- NO markdown
- NO commentary
- Do NOT cut output midway

JSON FORMAT (EXACT):

{{
  "chapter": "{ch['title']}",
  "easy": ["Q1","Q2","Q3","Q4","Q5"],
  "medium": ["Q1","Q2","Q3","Q4","Q5"],
  "hard": ["Q1","Q2","Q3","Q4","Q5"],
  "mcq": [
    {{
      "question": "string",
      "options": ["A", "B", "C", "D"],
      "answer": "A/B/C/D",
      "explanation": "string"
    }}
  ]
}}

REQUIREMENTS:
- 5 Easy
- 5 Medium
- 5 Hard
- ONLY 5 MCQs
- Ensure JSON ends correctly

CHAPTER CONTENT:
{ch['content'][:3000]}
"""

            q_raw = call_groq_chat(
                [{"role": "user", "content": q_prompt}],
                max_new_tokens=900
            )

            q_clean = re.sub(r"```json|```", "", q_raw).strip()

            # ---- SAFETY CHECK ----
            if not q_clean.endswith("}"):
                print("\n❌ INCOMPLETE QUESTION JSON SKIPPED\n")
                continue

            try:
                final_output.append(json.loads(q_clean))
            except:
                print("\n❌ BAD QUESTION JSON:\n", q_raw)
                continue

        return render_template("ai_questions.html", chapters=final_output)

    return render_template("ai_questions.html")


import os
import time
import json
import numpy as np
from uuid import uuid4

from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename

# --------------------------------------------------
# GLOBAL SESSION STORE (define ONCE)
# --------------------------------------------------
PDF_CHAT_SESSIONS = {}
# session_id : {
#   "chunks": [...],
#   "vectors": [...],
#   "history": [...],
#   "filepath": "...",
#   "meta": {...}
# }

# --------------------------------------------------
# CHAT PDF PAGE (UI ONLY)
# --------------------------------------------------
@app.route("/chat-pdf")
def chat_pdf_page():
    return render_template("chat_pdf.html")

# --------------------------------------------------
# PDF UPLOAD + SESSION CREATION (API)
# --------------------------------------------------
@app.route("/chat-pdf-upload", methods=["POST"])
def chat_pdf_upload():
    pdf = request.files.get("pdf")
    if not pdf:
        return jsonify({"error": "No file uploaded"}), 400

    # Secure filename
    filename = secure_filename(pdf.filename)
    stored_name = f"{uuid4().hex}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, stored_name)
    pdf.save(filepath)

    # Extract text
    text = extract_text_from_pdf(filepath).strip()
    if not text:
        return jsonify({"error": "No readable text in PDF"}), 400

    # Create session
    session_id = str(uuid4())

    # RAG chunking
    CHUNK_SIZE = 500 if len(text) < 1500 else 3000
    OVERLAP = 100

    chunks = chunk_text_chars(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    vectors = create_embeddings(chunks)

    PDF_CHAT_SESSIONS[session_id] = {
        "chunks": chunks,
        "vectors": vectors.tolist(),
        "history": [],
        "filepath": filepath,
        "meta": {
            "filename": filename,
            "created": time.time(),
            "default_style": "simple",
            "default_language": "english"
        }
    }

    # Small preview for frontend
    preview = text[:2000]

    return jsonify({
        "session_id": session_id,
        "filename": filename,
        "preview": preview
    })

@app.route("/chat-pdf-ask", methods=["POST"])
def chat_pdf_ask():
    data = request.json
    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or session_id not in PDF_CHAT_SESSIONS:
        return jsonify({"error": "Invalid session"}), 400

    session = PDF_CHAT_SESSIONS[session_id]
    chunks = session["chunks"]
    vectors = np.array(session["vectors"])

    # RAG retrieval
    context = "\n\n".join(retrieve_top_k(message, chunks, vectors))

    answer = call_groq_chat([
        {"role": "system", "content": "Answer strictly using the PDF context."},
        {"role": "user", "content": f"PDF Content:\n{context}\n\nQuestion: {message}"}
    ])

    return jsonify({"answer": answer})



from gtts import gTTS

def text_to_audio(text, output_path="output.mp3"):
    tts = gTTS(text=text, lang="en")
    tts.save(output_path)
    return output_path
from flask import send_from_directory

@app.route("/pdf-to-audio", methods=["POST"])
def pdf_to_audio():
    pdf = request.files.get("pdf")
    if not pdf:
        return {"error": "No PDF uploaded"}, 400

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)

    # Extract text
    text = extract_text_from_pdf(pdf_path)
    text = text[:4000]   # IMPORTANT: limit text
    if not text.strip():
        return {"error": "No readable text inside PDF"}, 400

    # Convert to audio
    audio_filename = pdf.filename.rsplit(".", 1)[0] + "_audio.mp3"
    audio_path = os.path.join(OUTPUT_FOLDER, audio_filename)

    tts = gTTS(text=text, lang="en")
    tts.save(audio_path)

    # Return audio file URL instead of download
    return {
        "audio_url": f"/outputs/{audio_filename}"
    }

@app.route("/pdf-to-audio-page")
def pdf_to_audio_page():
    return render_template("pdf_to_audio.html")
@app.route('/outputs/<path:filename>')
def serve_audio(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)