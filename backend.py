from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from google.genai import types
from PIL import Image
import io
import os
import json
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress ALTS credentials warning
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'

app = Flask(__name__)
# Enable CORS for the React frontend (running on a different port)
CORS(app)

# --- Configuration & Initialization ---
# Load environment variables from a .env file
load_dotenv()
# Get the API key from environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    client = None
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel('gemini-2.0-flash-lite')
        logger.info("Gemini Client Initialized Successfully.")
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {e}")
        client = None

# --- Helper Function for Multimodal Analysis ---
def analyze_with_gemini(file_stream, upload_type):
    if not client:
        logger.error("AI Service Not Available. Check API key setup.")
        return {"summary": "AI Service Not Available. Check API key setup.", "severity": "Error", "details": "N/A"}

    try:
        # Convert file stream (BytesIO) to PIL Image
        img = Image.open(file_stream)
        
        # 1. System Instruction (Guiding the AI's persona and output format)
        system_instruction = (
            "You are an expert, empathetic AI medical report analyzer. "
            "Analyze the uploaded document (X-ray, lab report, or prescription). "
            "Your output must be easy for a layperson to understand. "
            "Critically, your output must strictly adhere to the provided JSON schema. "
            "Do NOT include any text outside of the JSON object."
        )

        # 2. User Prompt based on content type
        if upload_type == 'medical_image':
            prompt = "Analyze this X-ray, CT, or medical image. Identify and describe any significant findings related to bone, tissue, or organs in simple, non-technical language. Provide a brief summary."
        elif upload_type == 'blood_report':
            prompt = "Analyze this lab report image. Identify all abnormal or out-of-range parameters. Explain what the deviation from the normal range might suggest in simple, simple terms."
        elif upload_type == 'prescription':
            prompt = (
                "Decode this prescription image. List all medications, their dosage, and the precise daily schedule/frequency. "
                "Place the decoded information in the 'details' field."
            )
        else:
            prompt = "Analyze this image and explain the medical content simply."

        # 3. Define the desired structured JSON output schema
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "summary": types.Schema(type=types.Type.STRING, description="A simplified, layman's summary of the findings."),
                "severity": types.Schema(type=types.Type.STRING, enum=['Low', 'Moderate', 'Severe', 'Informational'], description="Assessment of the finding's urgency or seriousness."),
                "details": types.Schema(type=types.Type.STRING, description="Detailed findings, e.g., abnormal values, decoded prescription, or specific diagnoses.")
            },
            required=["summary", "severity", "details"]
        )

        # 4. Generate Content (Multimodal Call)
        response = client.generate_content(
            contents=[
                system_instruction,
                img,
                prompt
            ],
            generation_config={
                "temperature": 0.2
            }
        )
        
        # Parse the response
        if response.text:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                return {
                    "summary": "Error parsing AI response",
                    "severity": "Error",
                    "details": response.text[:500]  # Include first 500 chars of response for debugging
                }

    except Exception as e:
        error_msg = f"Gemini API or processing failed: {str(e)}"
        logger.error(error_msg)
        return {"summary": error_msg, "severity": "Error", "details": "N/A"}

# --- API Route ---
@app.route('/api/analyze', methods=['POST'])
def analyze_report():
    logger.info("Received analysis request")
    file = request.files.get('file')
    upload_type = request.form.get('type')

    if not file:
        logger.warning("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    result = analyze_with_gemini(file.stream, upload_type)
    logger.info("Analysis completed successfully")
    return jsonify(result), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)