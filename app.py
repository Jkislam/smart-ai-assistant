from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import pytesseract
from PIL import Image
import base64
import io
import os
import fitz  # PyMuPDF
import json
import requests
from config import API_KEY
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re

# Initialize OpenAI client with new API
client = OpenAI(api_key=API_KEY)
app = Flask(__name__)
CORS(app)

# ===========================
# PDF Links লোড
# ===========================
with open("pdf_links.json", "r", encoding="utf-8") as f:
    pdf_links = json.load(f)

# ===========================
# নতুন ফিচার: কনভার্সেশন মেমরি
# ===========================
conversation_memory = {}

def get_conversation_history(user_id):
    return conversation_memory.get(user_id, [])

def update_conversation_history(user_id, role, content):
    if user_id not in conversation_memory:
        conversation_memory[user_id] = []
    
    # সর্বাধিক 10টি মেসেজ রাখা হবে
    if len(conversation_memory[user_id]) >= 10:
        conversation_memory[user_id].pop(0)
    
    conversation_memory[user_id].append({"role": role, "content": content})

# ===========================
# নতুন ফিচার: PDF টেক্সট এক্সট্রাক্টর
# ===========================
def extract_text_from_pdf_url(pdf_url):
    try:
        pdf_bytes = requests.get(pdf_url).content
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ===========================
# নতুন ফিচার: ইমেজ থেকে টেক্সট এক্সট্রাকশন
# ===========================
def extract_text_from_image(image_data):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        extracted_text = pytesseract.image_to_string(image, lang="eng+ben")
        return extracted_text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

# ===========================
# হোম রুট
# ===========================
@app.route('/')
def home():
    return jsonify({
        "message": "✅ Smart AI Helper API is Live",
        "version": "2.0",
        "python_version": "3.13",
        "endpoints": [
            {"path": "/summary", "method": "POST", "description": "Video/Text to Summary"},
            {"path": "/mcq", "method": "POST", "description": "Text to MCQ"},
            {"path": "/image-to-notes", "method": "POST", "description": "Image to Notes"},
            {"path": "/image-to-mcq", "method": "POST", "description": "Image to MCQ"},
            {"path": "/image-to-cq", "method": "POST", "description": "Image to Creative Questions"},
            {"path": "/routine", "method": "POST", "description": "Generate Study Routine"},
            {"path": "/chapter-to-mcq", "method": "POST", "description": "Chapter to MCQ from PDF"},
            {"path": "/chapter-to-cq", "method": "POST", "description": "Chapter to Creative Questions from PDF"},
            {"path": "/image-to-answer", "method": "POST", "description": "Image Question to Answer"},
            {"path": "/text-to-word-meaning", "method": "POST", "description": "Text to Word Meanings"},
            {"path": "/text-to-answer", "method": "POST", "description": "Text Question to Answer"},
            {"path": "/math-solver", "method": "POST", "description": "Math Problem Solver"},
            {"path": "/image-to-math-solver", "method": "POST", "description": "Image Math Problem to Solution"},
            {"path": "/chat", "method": "POST", "description": "AI Chat with Memory"},
            {"path": "/essay", "method": "POST", "description": "Essay Generator"},
            {"path": "/grammar-check", "method": "POST", "description": "Grammar and Spelling Check"},
            {"path": "/translate", "method": "POST", "description": "Text Translator"},
            {"path": "/flashcards", "method": "POST", "description": "Generate Flashcards"},
            {"path": "/homework-help", "method": "POST", "description": "Homework Assistance"},
            {"path": "/study-tips", "method": "POST", "description": "Study Tips Generator"}
        ]
    })

# ===========================
# (১) ভিডিও ➡️ সামারি
# ===========================
@app.route('/summary', methods=['POST'])
def summarize():
    data = request.json
    text = data.get("text", "")
    video_url = data.get("video_url", "")

    if video_url:
        try:
            parsed_url = urlparse(video_url)
            video_id = None
            if "youtube.com" in parsed_url.netloc and "v" in parse_qs(parsed_url.query):
                video_id = parse_qs(parsed_url.query).get("v")[0]
            elif "youtu.be" in parsed_url.netloc:
                video_id = parsed_url.path.lstrip("/")
            elif "youtube.com" in parsed_url.netloc and "/live/" in parsed_url.path:
                video_id = parsed_url.path.split("/live/")[1].split("?")[0]

            if video_id:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['bn', 'en'])
                    text = " ".join([entry['text'] for entry in transcript])
                except Exception:
                    return jsonify({"error": "Transcript not available for this video. Please enter text instead."}), 200
            else:
                return jsonify({"error": "Invalid YouTube URL"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    if not text:
        return jsonify({"error": "No text or video transcript provided"}), 400

    try:
        prompt = f"এই বক্তব্যটা সংক্ষেপে বাংলা ভাষায় বুঝিয়ে দাও:\n{text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"summary": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (২) টেক্সট ➡️ MCQ
# ===========================
@app.route('/mcq', methods=['POST'])
def mcq():
    data = request.json
    chapter = data.get("chapter", "")
    count = data.get("count", 5)
    difficulty = data.get("difficulty", "medium")
    
    if not chapter:
        return jsonify({"error": "Chapter text required"}), 400

    try:
        prompt = f"{chapter} বিষয় থেকে {count}টি {difficulty} difficulty MCQ তৈরি করো, অপশনসহ এবং সঠিক উত্তর দাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"mcqs": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৩) ছবি ➡️ নোট
# ===========================
@app.route('/image-to-notes', methods=['POST'])
def image_to_notes():
    data = request.json
    image_data = data.get("image_base64", "")
    if not image_data:
        return jsonify({"error": "Image data required"}), 400

    try:
        extracted_text = extract_text_from_image(image_data)
        
        if "Error extracting" in extracted_text:
            return jsonify({"error": extracted_text}), 400

        prompt = f"এই লেখাটার বাংলা ভাষায় সংক্ষিপ্ত নোট বানাও:\n{extracted_text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            "extracted_text": extracted_text,
            "summary": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৪) ছবি ➡️ MCQ
# ===========================
@app.route('/image-to-mcq', methods=['POST'])
def image_to_mcq():
    data = request.json
    image_data = data.get("image_base64", "")
    count = data.get("count", 5)
    
    if not image_data:
        return jsonify({"error": "Image data required"}), 400

    try:
        extracted_text = extract_text_from_image(image_data)
        
        if "Error extracting" in extracted_text:
            return jsonify({"error": extracted_text}), 400

        prompt = f"এই লেখাটার ভিত্তিতে {count}টি MCQ তৈরি করো:\n{extracted_text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            "extracted_text": extracted_text,
            "mcqs": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৫) ছবি ➡️ CQ
# ===========================
@app.route('/image-to-cq', methods=['POST'])
def image_to_cq():
    data = request.json
    image_data = data.get("image_base64", "")
    
    if not image_data:
        return jsonify({"error": "Image data required"}), 400

    try:
        extracted_text = extract_text_from_image(image_data)
        
        if "Error extracting" in extracted_text:
            return jsonify({"error": extracted_text}), 400

        prompt = f"এই লেখাটার ভিত্তিতে একটি সৃজনশীল প্রশ্ন বানাও:\n{extracted_text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            "extracted_text": extracted_text,
            "cq": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৬) রুটিন
# ===========================
@app.route('/routine', methods=['POST'])
def routine():
    data = request.json
    subjects = data.get("subjects", "")
    hours = data.get("hours", 2)
    days = data.get("days", 7)

    if not subjects:
        return jsonify({"error": "Subjects required"}), 400

    try:
        prompt = f"এই বিষয়গুলো: {subjects} দিয়ে প্রতিদিন {hours} ঘন্টা করে {days} দিনের পড়াশোনার রুটিন বানাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"routine": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৭) অধ্যায় ➡️ MCQ (PDF থেকে)
# ===========================
@app.route('/chapter-to-mcq', methods=['POST'])
def chapter_to_mcq():
    data = request.json
    class_name = data.get("class")
    subject = data.get("subject")
    chapter = data.get("chapter")
    count = data.get("count", 5)

    if not (class_name and subject and chapter):
        return jsonify({"error": "class, subject, chapter required"}), 400

    try:
        if class_name not in pdf_links or subject not in pdf_links[class_name]:
            return jsonify({"error": "PDF not available for this class/subject"}), 404
            
        pdf_url = pdf_links[class_name][subject]
        text = extract_text_from_pdf_url(pdf_url)
        
        if "Error extracting" in text:
            return jsonify({"error": text}), 400

        prompt = f"অধ্যায়: {chapter}\nএই লেখা থেকে {count}টি MCQ তৈরি করো:\n\n{text[:4000]}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"mcqs": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৮) অধ্যায় ➡️ CQ (PDF থেকে)
# ===========================
@app.route('/chapter-to-cq', methods=['POST'])
def chapter_to_cq():
    data = request.json
    class_name = data.get("class")
    subject = data.get("subject")
    chapter = data.get("chapter")
    count = data.get("count", 2)

    if not (class_name and subject and chapter):
        return jsonify({"error": "class, subject, chapter required"}), 400

    try:
        if class_name not in pdf_links or subject not in pdf_links[class_name]:
            return jsonify({"error": "PDF not available for this class/subject"}), 404
            
        pdf_url = pdf_links[class_name][subject]
        text = extract_text_from_pdf_url(pdf_url)
        
        if "Error extracting" in text:
            return jsonify({"error": text}), 400

        prompt = f"অধ্যায়: {chapter}\nএই লেখা থেকে {count}টি CQ তৈরি করো:\n\n{text[:4000]}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"cqs": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (৯) ছবি ➡️ উত্তর
# ===========================
@app.route('/image-to-answer', methods=['POST'])
def image_to_answer():
    data = request.json
    image_data = data.get("image_base64", "")
    
    if not image_data:
        return jsonify({"error": "Image data required"}), 400

    try:
        extracted_text = extract_text_from_image(image_data)
        
        if "Error extracting" in extracted_text:
            return jsonify({"error": extracted_text}), 400

        prompt = f"প্রশ্ন: {extracted_text}\nএটির সঠিক উত্তর দাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            "extracted_text": extracted_text,
            "answer": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (১০) টেক্সট ➡️ শব্দার্থ
# ===========================
@app.route('/text-to-word-meaning', methods=['POST'])
def text_to_word_meaning():
    data = request.json
    text = data.get("text", "")
    language = data.get("language", "bangla")
    
    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        prompt = f"এই শব্দগুলোর {language} অর্থ দাও:\n{text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"word_meanings": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (১১) টেক্সট ➡️ উত্তর
# ===========================
@app.route('/text-to-answer', methods=['POST'])
def text_to_answer():
    data = request.json
    question = data.get("question", "")
    language = data.get("language", "bangla")
    
    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        prompt = f"প্রশ্ন: {question}\nএটির সঠিক উত্তর {language} ভাষায় দাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (১২) ম্যাথ সলভার
# ===========================
@app.route('/math-solver', methods=['POST'])
def math_solver():
    data = request.json
    math_problem = data.get("problem", "")
    language = data.get("language", "bangla")
    
    if not math_problem:
        return jsonify({"error": "Problem required"}), 400

    try:
        prompt = f"সমস্যা: {math_problem}\nএটি ধাপে ধাপে {language} ভাষায় সমাধান করো।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"solution": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# (১৩) ছবি ➡️ ম্যাথ সলভর
# ===========================
@app.route('/image-to-math-solver', methods=['POST'])
def image_to_math_solver():
    data = request.json
    image_data = data.get("image_base64", "")
    language = data.get("language", "bangla")
    
    if not image_data:
        return jsonify({"error": "Image data required"}), 400

    try:
        math_text = extract_text_from_image(image_data)
        
        if "Error extracting" in math_text:
            return jsonify({"error": math_text}), 400

        prompt = f"সমস্যা: {math_text}\nএটি ধাপে ধাপে {language} ভাষায় সমাধান করো।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({
            "extracted_text": math_text,
            "solution": response.choices[0].message.content
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: AI চ্যাট (মেমরি সহ)
# ===========================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get("message", "")
    user_id = data.get("user_id", "default_user")
    language = data.get("language", "bangla")
    
    if not message:
        return jsonify({"error": "Message required"}), 400

    try:
        # পূর্বের কথোপকথন লোড করুন
        history = get_conversation_history(user_id)
        
        # সিস্টেম প্রম্পট
        system_prompt = {
            "role": "system", 
            "content": f"তুমি একজন শিক্ষার্থীদের জন্য সহায়ক AI। {language} ভাষায় উত্তর দাও।"
        }
        
        # বর্তমান মেসেজ
        user_message = {"role": "user", "content": message}
        
        # সম্পূর্ণ কথোপকথন তৈরি করুন
        messages = [system_prompt] + history + [user_message]
        
        # OpenAI API কল করুন (Updated for 1.x+)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        ai_response = response.choices[0].message.content
        
        # কথোপকথন আপডেট করুন
        update_conversation_history(user_id, "user", message)
        update_conversation_history(user_id, "assistant", ai_response)
        
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: প্রবন্ধ জেনারেটর
# ===========================
@app.route('/essay', methods=['POST'])
def essay_generator():
    data = request.json
    topic = data.get("topic", "")
    word_count = data.get("word_count", 300)
    language = data.get("language", "bangla")
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400

    try:
        prompt = f"{topic} সম্পর্কে {word_count} শব্দের একটি প্রবন্ধ {language} ভাষায় লিখ।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"essay": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: ব্যাকরণ ও বানান পরীক্ষক
# ===========================
@app.route('/grammar-check', methods=['POST'])
def grammar_check():
    data = request.json
    text = data.get("text", "")
    language = data.get("language", "bangla")
    
    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        prompt = f"এই লেখাটির ব্যাকরণ ও بানান সংশোধন করে {language} ভাষায় দেখাও:\n{text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"corrected_text": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: অনুবাদক
# ===========================
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    from_lang = data.get("from_lang", "english")
    to_lang = data.get("to_lang", "bangla")
    
    if not text:
        return jsonify({"error": "Text required"}), 400

    try:
        prompt = f"এই লেখাটি {from_lang} থেকে {to_lang} ভাষায় অনুবাদ কর:\n{text}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"translated_text": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: ফ্ল্যাশকার্ড জেনারেটর
# ===========================
@app.route('/flashcards', methods=['POST'])
def flashcards():
    data = request.json
    topic = data.get("topic", "")
    count = data.get("count", 10)
    language = data.get("language", "bangla")
    
    if not topic:
        return jsonify({"error": "Topic required"}), 400

    try:
        prompt = f"{topic} সম্পর্কে {count}টি ফ্ল্যাশকার্ড তৈরি করো (প্রশ্ন এবং উত্তর সহ)। {language} ভাষায় উত্তর দাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"flashcards": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: হোমওয়ার্ক হেল্প
# ===========================
@app.route('/homework-help', methods=['POST'])
def homework_help():
    data = request.json
    question = data.get("question", "")
    subject = data.get("subject", "")
    language = data.get("language", "bangla")
    
    if not question:
        return jsonify({"error": "Question required"}), 400

    try:
        prompt = f"{subject} বিষয়ে এই হোমওয়ার্ক প্রশ্নের উত্তর {language} ভাষায় দাও:\n{question}"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# নতুন ফিচার: পড়ার টিপস
# ===========================
@app.route('/study-tips', methods=['POST'])
def study_tips():
    data = request.json
    subject = data.get("subject", "")
    topic = data.get("topic", "")
    language = data.get("language", "bangla")
    
    try:
        prompt = f"{subject} - {topic} পড়ার জন্য কার্যকরী টিপস {language} ভাষায় দাও।"
        
        # Updated for OpenAI 1.x+
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return jsonify({"study_tips": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================
# রান
# ===========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=True)