# main.py
import os
import time
import json
import traceback
import asyncio
import requests
import re
import base64
from typing import Optional, Any, Dict, List
from urllib.parse import urljoin, urlparse
from io import StringIO, BytesIO

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright

# Optional libraries
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image
except ImportError:
    Image = None

# Environment variables
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL")
STUDENT_SECRET = os.getenv("STUDENT_SECRET")

if not STUDENT_EMAIL or not STUDENT_SECRET:
    raise RuntimeError("Set STUDENT_EMAIL and STUDENT_SECRET environment variables.")

TOTAL_TIMEOUT_SECONDS = 170  # Leave 10s buffer from 3min limit

app = FastAPI()


# ============================================================================
# Helper Functions
# ============================================================================

async def parse_raw_json(request: Request) -> dict:
    """Parse JSON from request body with error handling."""
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")


def download_file(url: str, timeout: int = 30) -> Optional[bytes]:
    """Download file from URL and return bytes."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"[Download] Failed to download {url}: {e}")
        return None


# ============================================================================
# Data Processing Functions
# ============================================================================

def process_csv(content: bytes) -> Optional[pd.DataFrame]:
    """Process CSV content and return DataFrame."""
    if not pd:
        return None
    try:
        # Try UTF-8 first
        try:
            df = pd.read_csv(BytesIO(content))
        except:
            df = pd.read_csv(BytesIO(content), encoding='latin1')
        return df
    except Exception as e:
        print(f"[CSV] Processing error: {e}")
        return None


def process_excel(content: bytes) -> Optional[pd.DataFrame]:
    """Process Excel content and return DataFrame."""
    if not pd:
        return None
    try:
        df = pd.read_excel(BytesIO(content))
        return df
    except Exception as e:
        print(f"[Excel] Processing error: {e}")
        return None


def process_pdf(content: bytes) -> Dict[str, Any]:
    """Extract text and tables from PDF."""
    result = {"text": "", "tables": []}
    
    if not pdfplumber:
        return result
    
    try:
        with pdfplumber.open(BytesIO(content)) as pdf:
            # Extract text from all pages
            texts = []
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
            result["text"] = "\n".join(texts)
            
            # Extract tables
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        result["tables"].append(table)
        
        return result
    except Exception as e:
        print(f"[PDF] Processing error: {e}")
        return result


def sum_column(df: pd.DataFrame, col_name: str = "value") -> Optional[float]:
    """Sum a column from DataFrame (case-insensitive match)."""
    if df is None or df.empty:
        return None
    
    # Find matching column
    cols = [c for c in df.columns if c.strip().lower() == col_name.lower()]
    if not cols:
        # Try partial match
        cols = [c for c in df.columns if col_name.lower() in c.strip().lower()]
    
    if not cols:
        return None
    
    try:
        total = pd.to_numeric(df[cols[0]], errors='coerce').fillna(0).sum()
        return float(total)
    except Exception as e:
        print(f"[Sum] Error summing column: {e}")
        return None


def count_rows(df: pd.DataFrame) -> Optional[int]:
    """Count rows in DataFrame."""
    if df is None:
        return None
    return len(df)


def get_max_value(df: pd.DataFrame, col_name: str) -> Optional[float]:
    """Get maximum value from column."""
    if df is None or df.empty:
        return None
    
    cols = [c for c in df.columns if col_name.lower() in c.strip().lower()]
    if not cols:
        return None
    
    try:
        return float(df[cols[0]].max())
    except:
        return None


def create_chart_base64(df: pd.DataFrame, chart_type: str = "bar") -> Optional[str]:
    """Create a simple chart and return as base64 data URI."""
    if not plt or df is None or df.empty:
        return None
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if chart_type == "bar" and len(df.columns) >= 2:
            df.plot(kind='bar', x=df.columns[0], y=df.columns[1], ax=ax)
        elif chart_type == "line" and len(df.columns) >= 2:
            df.plot(kind='line', x=df.columns[0], y=df.columns[1], ax=ax)
        else:
            # Default: plot first numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                df[numeric_cols[0]].plot(kind='bar', ax=ax)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"[Chart] Error creating chart: {e}")
        return None


# ============================================================================
# Question Analysis & Answer Computation
# ============================================================================

def analyze_question(question: str) -> Dict[str, Any]:
    """Analyze question text to determine what's being asked."""
    if not question:
        return {}
    
    q_lower = question.lower()
    analysis = {
        "type": "unknown",
        "operation": None,
        "column": None,
        "needs_viz": False
    }
    
    # Detect operation type
    if any(word in q_lower for word in ["sum", "total", "add"]):
        analysis["operation"] = "sum"
    elif any(word in q_lower for word in ["count", "how many", "number of"]):
        analysis["operation"] = "count"
    elif any(word in q_lower for word in ["average", "mean", "avg"]):
        analysis["operation"] = "average"
    elif any(word in q_lower for word in ["max", "maximum", "highest", "largest"]):
        analysis["operation"] = "max"
    elif any(word in q_lower for word in ["min", "minimum", "lowest", "smallest"]):
        analysis["operation"] = "min"
    
    # Detect if visualization needed
    if any(word in q_lower for word in ["chart", "graph", "plot", "visualize", "visualization"]):
        analysis["needs_viz"] = True
    
    # Extract column name if mentioned
    value_match = re.search(r'"([^"]+)"\s*column', q_lower)
    if value_match:
        analysis["column"] = value_match.group(1)
    
    return analysis


async def compute_answer(page_data: Dict[str, Any], question: str) -> Any:
    """Compute answer based on page data and question."""
    
    # Analyze question
    q_analysis = analyze_question(question)
    print(f"[Compute] Question analysis: {q_analysis}")
    
    answer = None
    data_sources = page_data.get("data_sources", [])
    
    # Process data sources
    for source_url in data_sources:
        content = download_file(source_url)
        if not content:
            continue
        
        df = None
        
        # Determine file type and process
        if source_url.lower().endswith('.csv'):
            df = process_csv(content)
        elif source_url.lower().endswith(('.xlsx', '.xls')):
            df = process_excel(content)
        elif source_url.lower().endswith('.pdf'):
            pdf_data = process_pdf(content)
            # Try to convert PDF table to DataFrame
            if pdf_data["tables"]:
                for table in pdf_data["tables"]:
                    try:
                        if pd:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            break
                    except:
                        continue
        
        # Compute based on operation
        if df is not None and not df.empty:
            col_name = q_analysis.get("column") or "value"
            
            if q_analysis["operation"] == "sum":
                result = sum_column(df, col_name)
                if result is not None:
                    answer = result
                    break
            elif q_analysis["operation"] == "count":
                answer = count_rows(df)
                break
            elif q_analysis["operation"] == "max":
                result = get_max_value(df, col_name)
                if result is not None:
                    answer = result
                    break
            elif q_analysis["needs_viz"]:
                chart = create_chart_base64(df)
                if chart:
                    answer = chart
                    break
    
    # Check for inline HTML tables
    if answer is None and page_data.get("html_table"):
        df = page_data["html_table"]
        col_name = q_analysis.get("column") or "value"
        
        if q_analysis["operation"] == "sum":
            answer = sum_column(df, col_name)
        elif q_analysis["operation"] == "count":
            answer = count_rows(df)
    
    # Check embedded data
    if answer is None and page_data.get("embedded_data"):
        answer = page_data["embedded_data"]
    
    # Fallback for yes/no questions
    if answer is None and question:
        q_lower = question.lower()
        if q_lower.startswith(("is ", "are ", "does ", "do ", "can ", "will ")):
            answer = True
    
    return answer


# ============================================================================
# Page Scraping
# ============================================================================

async def scrape_quiz_page(url: str) -> Dict[str, Any]:
    """Scrape quiz page and extract all relevant information."""
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        page = await browser.new_page()
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=60000)
            await asyncio.sleep(1)  # Let JS finish
            
            result = {
                "question": None,
                "submit_url": None,
                "data_sources": [],
                "html_table": None,
                "embedded_data": None
            }
            
            # Extract question
            for selector in ["h1", "h2", ".question", "#question", "#result"]:
                elem = await page.query_selector(selector)
                if elem:
                    text = await elem.inner_text()
                    if text and len(text.strip()) > 10:
                        result["question"] = text.strip()
                        break
            
            # Extract submit URL
            # Check <pre> for JSON first
            pre_elem = await page.query_selector("pre")
            if pre_elem:
                try:
                    pre_text = await pre_elem.inner_text()
                    data = json.loads(pre_text)
                    result["submit_url"] = data.get("submit_url")
                    if data.get("question"):
                        result["question"] = data["question"]
                except:
                    pass
            
            # Check form action
            if not result["submit_url"]:
                form = await page.query_selector("form")
                if form:
                    action = await form.get_attribute("action")
                    if action:
                        result["submit_url"] = urljoin(url, action)
            
            # Search HTML for submit URL pattern
            if not result["submit_url"]:
                html = await page.content()
                match = re.search(r'(https?://[^\s"\'<>]+/submit[^\s"\'<>]*)', html)
                if match:
                    result["submit_url"] = match.group(1)
            
            # Extract data source links
            links = await page.query_selector_all("a")
            for link in links:
                href = await link.get_attribute("href")
                if href:
                    full_url = urljoin(url, href)
                    if any(full_url.lower().endswith(ext) for ext in ['.csv', '.pdf', '.xlsx', '.xls', '.json']):
                        result["data_sources"].append(full_url)
            
            # Extract HTML tables
            table = await page.query_selector("table")
            if table and pd:
                rows = await table.query_selector_all("tr")
                if len(rows) >= 2:
                    # Extract headers
                    header_cells = await rows[0].query_selector_all("th, td")
                    headers = [await c.inner_text() for c in header_cells]
                    
                    # Extract data rows
                    data_rows = []
                    for row in rows[1:]:
                        cells = await row.query_selector_all("td")
                        data_rows.append([await c.inner_text() for c in cells])
                    
                    try:
                        result["html_table"] = pd.DataFrame(data_rows, columns=headers)
                    except:
                        pass
            
            # Check for embedded base64 data
            html = await page.content()
            atob_matches = re.findall(r'atob\([`\'"]([^`\'"]+)[`\'"]\)', html)
            for encoded in atob_matches:
                try:
                    decoded = base64.b64decode(encoded).decode('utf-8')
                    # Try parsing as JSON
                    try:
                        data = json.loads(decoded)
                        if isinstance(data, dict) and "answer" in data:
                            result["embedded_data"] = data["answer"]
                            break
                    except:
                        # Extract numbers from decoded text
                        numbers = re.findall(r'\d+(?:\.\d+)?', decoded)
                        if numbers:
                            result["embedded_data"] = float(numbers[0])
                except:
                    continue
            
            return result
            
        finally:
            await page.close()
            await browser.close()


# ============================================================================
# Main Solver Logic
# ============================================================================

async def solve_quiz_chain(start_url: str, email: str, secret: str):
    """Main quiz solving loop."""
    start_time = time.time()
    current_url = start_url
    attempts = 0
    max_attempts = 20  # Prevent infinite loops
    
    print(f"[Solver] Starting quiz chain from: {start_url}")
    
    while attempts < max_attempts:
        attempts += 1
        elapsed = time.time() - start_time
        
        if elapsed > TOTAL_TIMEOUT_SECONDS:
            print(f"[Solver] Timeout reached after {elapsed:.1f}s")
            return
        
        print(f"\n[Solver] Attempt {attempts} - URL: {current_url}")
        
        try:
            # Scrape the page
            page_data = await scrape_quiz_page(current_url)
            print(f"[Solver] Page data: question={page_data['question'][:50] if page_data['question'] else None}...")
            print(f"[Solver] Submit URL: {page_data['submit_url']}")
            print(f"[Solver] Data sources: {page_data['data_sources']}")
            
            if not page_data["submit_url"]:
                print("[Solver] No submit URL found, aborting")
                return
            
            # Compute answer
            answer = await compute_answer(page_data, page_data["question"] or "")
            print(f"[Solver] Computed answer: {answer}")
            
            # Submit answer
            payload = {
                "email": email,
                "secret": secret,
                "url": current_url,
                "answer": answer
            }
            
            print(f"[Solver] Submitting to: {page_data['submit_url']}")
            response = requests.post(page_data["submit_url"], json=payload, timeout=30)
            
            print(f"[Solver] Response status: {response.status_code}")
            
            try:
                resp_data = response.json()
                print(f"[Solver] Response data: {resp_data}")
            except:
                print(f"[Solver] Response text: {response.text[:200]}")
                return
            
            # Check if correct
            if resp_data.get("correct"):
                print("[Solver] ✓ Answer correct!")
                next_url = resp_data.get("url")
                
                if not next_url:
                    print("[Solver] Quiz completed successfully!")
                    return
                
                current_url = urljoin(current_url, next_url)
                print(f"[Solver] Moving to next question: {current_url}")
                
            else:
                reason = resp_data.get("reason", "Unknown")
                print(f"[Solver] ✗ Answer incorrect: {reason}")
                
                # Check if we got a next URL anyway
                next_url = resp_data.get("url")
                if next_url:
                    print(f"[Solver] Skipping to next question: {next_url}")
                    current_url = urljoin(current_url, next_url)
                else:
                    print("[Solver] No next URL, stopping")
                    return
            
        except Exception as e:
            print(f"[Solver] Error: {e}")
            print(traceback.format_exc())
            return
    
    print(f"[Solver] Max attempts ({max_attempts}) reached")


# ============================================================================
# FastAPI Endpoints
# ============================================================================

@app.post("/")
async def handle_quiz(request: Request):
    """Main endpoint to receive quiz requests."""
    payload = await parse_raw_json(request)
    
    # Validate secret
    if payload.get("secret") != STUDENT_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    
    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")
    
    if not all([email, secret, url]):
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    # Start solver in background
    asyncio.create_task(solve_quiz_chain(url, email, secret))
    
    return JSONResponse({"status": "accepted", "message": "Quiz solving started"})


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "status": "running",
        "email": STUDENT_EMAIL,
        "libraries": {
            "pandas": pd is not None,
            "pdfplumber": pdfplumber is not None,
            "matplotlib": plt is not None
        }
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}
