import streamlit as st
from groq import Groq
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import io
import json
from typing import Optional

# ----------------------------------------------------------------------
# Text extraction from PDF (digital text first, OCR fallback)
# ----------------------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Try to extract text using pypdf (fast). If result is too short,
    fall back to OCR using pdf2image + pytesseract.
    """
    # 1) Try digital text extraction
    reader = PdfReader(io.BytesIO(pdf_bytes))
    digital_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            digital_text += text + "\n"

    # If we got a reasonable amount of text, return it
    if len(digital_text.strip()) > 500:
        return digital_text.strip()

    # 2) Fallback to OCR (slower but works for scanned documents)
    st.info("Digital text limited – using OCR (this may take a while)...")
    images = convert_from_bytes(pdf_bytes, dpi=200)
    ocr_text = ""
    # Use French language for better recognition
    custom_config = r'--oem 3 --psm 6 -l fra'
    for i, img in enumerate(images):
        st.write(f"OCR processing page {i+1}/{len(images)}...")
        text = pytesseract.image_to_string(img, config=custom_config)
        ocr_text += text + "\n"
    return ocr_text.strip()

# ----------------------------------------------------------------------
# Build prompt for Groq (same as before)
# ----------------------------------------------------------------------
def build_prompt(extracted_text: str) -> str:
    """
    Create a detailed prompt instructing the LLM to extract French tax data into JSON.
    """
    return f"""
Tu es un expert en comptabilité et fiscalité française, spécialisé dans les liasses fiscales (séries 2050, 2051, 2052, 2053, 2054, etc.).

Voici le texte extrait d'une liasse fiscale (probablement formulaire n° 2050 et annexes). 
Extrait toutes les informations financières et fiscales pertinentes suivantes, et retourne-les **uniquement** sous forme d'un JSON valide.

Le JSON doit contenir (si disponible) les champs suivants, avec des nombres sous forme de nombres (pas de chaînes) :

{{
  "company_name": "nom de l'entreprise",
  "siret": "numéro SIRET",
  "fiscal_year_end": "date de clôture de l'exercice (AAAA-MM-JJ)",
  "balance_sheet": {{
    "total_assets": 1234567,
    "total_liabilities": 1234567,
    "shareholders_equity": 123456,
    "provisions": 12345,
    "debts": 987654
  }},
  "income_statement": {{
    "turnover_net": 2345678,
    "operating_profit": 123456,
    "financial_result": 7890,
    "net_profit_loss": 54321
  }},
  "tax_result": {{
    "taxable_profit": 50000,
    "tax_loss_carryforward": 0,
    "corporate_income_tax": 15000
  }},
  "other_figures": {{
    "value_added": 800000,
    "staff_count": 12
  }}
}}

Si certaines données ne sont pas trouvées, mets `null`. Ne rajoute pas d'autres commentaires, ne répète pas le prompt, renvoie uniquement le JSON.

Texte à analyser :
----------------------------------------
{extracted_text}
----------------------------------------
"""

# ----------------------------------------------------------------------
# Call Groq API
# ----------------------------------------------------------------------
def call_groq(api_key: str, prompt: str) -> Optional[str]:
    try:
        # Initialize the Groq client with your API key
        client = Groq(api_key=api_key)
        
        # Make the API call
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # You can change this model. See Groq console for options.
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.3,
            # Instruct the model to respond with a JSON object
            response_format={"type": "json_object"},
        )
        
        # Extract the response content
        if chat_completion and chat_completion.choices:
            return chat_completion.choices[0].message.content.strip()
        else:
            st.error("Groq returned an empty response.")
            return None
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None

# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="French Tax Liasse OCR + Groq", layout="wide")
    st.title("📄 French Tax Liasse (2050 series) → JSON using OCR + Groq")
    st.markdown("Upload a PDF file (scan or digital) of the French tax forms. The app will extract text (with OCR fallback) and ask Groq's LLM to produce a structured JSON.")

    # Safely load API key from secrets (if any)
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("API key loaded from secrets.")
        else:
            st.warning("secrets.toml found but no GROQ_API_KEY inside.")
    except FileNotFoundError:
        st.info("No secrets.toml found. Please enter your API key manually.")

    # If not in secrets, ask user to input
    if not api_key:
        api_key = st.text_input("Your Groq API Key:", type="password")
        if not api_key:
            st.warning("Please enter your Groq API key to continue.")
            st.stop()

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        with st.spinner("Extracting text from PDF (digital or OCR)..."):
            extracted_text = extract_text_from_pdf(pdf_bytes)

        if not extracted_text:
            st.error("No text could be extracted from the PDF.")
            st.stop()

        with st.expander("Show extracted text (first 2000 characters)"):
            st.text(extracted_text[:2000])

        st.info("Sending to Groq for JSON extraction...")
        prompt = build_prompt(extracted_text)
        json_response = call_groq(api_key, prompt)

        if json_response:
            try:
                data = json.loads(json_response)
                pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
                st.success("✅ Extraction successful!")
                st.subheader("Extracted JSON")
                st.json(data)

                st.download_button(
                    label="📥 Download JSON file",
                    data=pretty_json,
                    file_name="liasse_extracted.json",
                    mime="application/json"
                )
            except json.JSONDecodeError:
                st.error("Groq did not return valid JSON. Raw response:")
                st.code(json_response)
        else:
            st.error("No response from Groq.")

if __name__ == "__main__":
    main()