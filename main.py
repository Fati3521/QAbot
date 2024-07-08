from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import fitz  # PyMuPDF
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = FastAPI()

# Charger le modèle T5 et le tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def extract_text_from_pdf(document):
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    document.close()
    return text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text

def segment_text_by_paragraph(text):
    paragraphs = text.split('\n\n')
    return paragraphs

def find_answer_in_segments(question, segments):
    best_answer = ""
    best_score = float('-inf')
    for segment in segments:
        input_text = f"question: {question} context: {segment}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(input_ids, max_length=150, num_beams=3, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score = len(answer)
        if score > best_score:
            best_score = score
            best_answer = answer
    return best_answer

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <body>
            <h1>Question Answering Bot</h1>
            <form action="/uploadfile/" enctype="multipart/form-data" method="post">
            <input name="file" type="file">
            <input name="question" type="text" placeholder="Enter your question here...">
            <input type="submit">
            </form>
        </body>
    </html>
    """

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...), question: str = Form(...)):
    contents = await file.read()
    document = fitz.open(stream=contents, filetype="pdf")
    text = extract_text_from_pdf(document)
    cleaned_text = clean_text(text)
    segments = segment_text_by_paragraph(cleaned_text)
    answer = find_answer_in_segments(question, segments)
    return {"Question": question, "Answer": answer}
