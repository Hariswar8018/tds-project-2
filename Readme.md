# TDS LLM Analysis Quiz - Backend

This is a FastAPI backend that solves multi-step quiz tasks for the TDS LLM Analysis project using:

- FastAPI
- Playwright (Chromium)
- Gemini (LLM)
- Python data processing tools

The backend receives a quiz URL, solves the quiz in the background, and returns a 200 OK immediately.

---
##🧠 Architecture
Client (Evaluator)
        ↓ POST / 
 FastAPI Backend
        ↓ 
 Background Worker
        ↓ 
Fetch Quiz Page (Playwright)
        ↓ 
Extract Question + Submit URL (LLM)
        ↓ 
Solve Question (Python)
        ↓ 
POST Answer to Submit URL
        ↓ 
Follow Next Quiz URL (loop)

---

## 🚀 Features
- Loads JS-rendered quiz pages using Playwright
- Uses Gemini only to extract question + submit URL
- Solves quizzes using Python
- Follows multi-step quiz chains
- Submits answers within the 3-minute limit
- Safe: no secrets in GitHub (uses environment variables)

---

## 📁 Project Structure

tds-project-2/
│
├── main.py                 # FastAPI server + background worker
├── debug_models.py         # List Gemini models
├── test_llm.py             # Test Gemini connectivity
├── test_server.py          # Test server connectivity
│
├── requirements.txt
├── render.yaml             # Deployment config for Render
├── LICENSE                 # MIT License
├── README.md
├── .env.example            # Example environment variables
└── .gitignore

## ⚙️ Environment Variables
Create a `.env` file locally (DO NOT commit it):
GEMINI_API_KEY=your_key_here
STUDENT_EMAIL=24f2008611@ds.study.iitm.ac.in
STUDENT_SECRET=your_secret_here
TEST_SERVER_URL=https://your-render-url.onrender.com

## ▶️ Running Locally
pip install -r requirements.txt
playwright install
uvicorn main:app --reload

Visit:
http://127.0.0.1:8000/

## 🌐 Deployment (Render)
Render detects `render.yaml` automatically.

Add these environment variables in Render:
GEMINI_API_KEY
STUDENT_EMAIL
STUDENT_SECRET

Your final Render URL will look like:
https://tds-llm-backend.onrender.com

## 🧪 Testing
python test_llm.py
python test_server.py

## 📜 License
MIT License.
