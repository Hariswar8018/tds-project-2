import os
import requests
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

TEST_SERVER_URL = os.getenv("TEST_SERVER_URL")

if not TEST_SERVER_URL:
    print("❌ TEST_SERVER_URL is not set in environment variables.")
    print("Add TEST_SERVER_URL to your .env file.")
    exit()

payload = {
    "email": os.getenv("STUDENT_EMAIL"),
    "secret": os.getenv("STUDENT_SECRET"),
    "url": "https://tds-llm-analysis.s-anand.net/demo"
}

print(f"Sending POST request to: {TEST_SERVER_URL}/run")

try:
    response = requests.post(f"{TEST_SERVER_URL}/run", json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.text)

    if response.status_code == 200:
        print("✅ TEST PASSED: Server accepted the request.")
    else:
        print("❌ TEST FAILED: Server rejected the request.")

except Exception as e:
    print("❌ CONNECTION ERROR:", e)