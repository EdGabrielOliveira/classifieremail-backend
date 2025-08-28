from fastapi import FastAPI, Header, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import pdfplumber
import fitz
from PIL import Image
import pytesseract
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from typing import Dict, Any
import io
import subprocess
from nlp_processor import EmailNLPProcessor

load_dotenv()

app = FastAPI(
    title="ClassifierEmail API",
    description="API para classificação inteligente de emails com NLP e IA",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("SECRET_API_KEY")
MODEL_AI_API = os.getenv("MODEL_AI_API")
OPENROUNTER_URL = os.getenv("OPENROUNTER_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

nlp_processor = EmailNLPProcessor()

class EmailDTO(BaseModel):
    remetente: str
    assunto: str
    descricao: str

class EmailResponse(BaseModel):
    classificacao: str
    rating: str
    motivo: str
    resposta_sugerida: str
    nlp_features: Dict[str, Any]
    quality_score: float

def parse_ai_response(ai_response: str) -> Dict[str, str]:
    try:
        lines = ai_response.strip().split('\n')
        result = {
            "classificacao": "Não classificado",
            "rating": "0/10",
            "motivo": "Resposta inválida da IA",
            "resposta_sugerida": "Obrigado pelo contato."
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith("CLASSIFICACAO:"):
                result["classificacao"] = line.replace("CLASSIFICACAO:", "").strip()
            elif line.startswith("RATING:"):
                result["rating"] = line.replace("RATING:", "").strip()
            elif line.startswith("MOTIVO:"):
                result["motivo"] = line.replace("MOTIVO:", "").strip()
            elif line.startswith("RESPOSTA_SUGERIDA:"):
                result["resposta_sugerida"] = line.replace("RESPOSTA_SUGERIDA:", "").strip()
        
        return result
    except Exception:
        return {
            "classificacao": "Erro",
            "rating": "0/10",
            "motivo": "Erro ao processar resposta da IA",
            "resposta_sugerida": "Obrigado pelo contato."
        }

def call_ai_api(content: str, nlp_data: Dict) -> Dict[str, str]:
    with open("diretrizes.txt", "r", encoding="utf-8") as f:
        diretrizes = f.read()
    
    nlp_context = f"""
    Dados NLP processados:
    - Texto limpo: {nlp_data['cleaned']['full_text']}
    - Tokens principais: {', '.join(nlp_data['cleaned']['tokens'][:10])}
    - Características: Palavras: {nlp_data['features']['word_count']}, Sentenças: {nlp_data['features']['sentence_count']}, Erros: {nlp_data['features']['spelling_errors']}
    - Score de qualidade: {nlp_processor.get_text_quality_score(nlp_data['features'])}/10
    
    Conteúdo original para análise:
    {content}
    """
    
    url = f"{OPENROUNTER_URL}"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": f"{MODEL_AI_API}",
        "messages": [
            {"role": "system", "content": diretrizes},
            {"role": "user", "content": nlp_context}
        ],
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]
        return parse_ai_response(ai_response)
    except Exception as e:
        return {
            "classificacao": "Erro",
            "rating": "0/10", 
            "motivo": f"Erro na API: {str(e)}",
            "resposta_sugerida": "Obrigado pelo contato."
        }

@app.post("/classemail", response_model=EmailResponse)
def analise_email(email: EmailDTO, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de autenticação inválida")
    
    nlp_data = nlp_processor.process_email_content(
        email.remetente, 
        email.assunto, 
        email.descricao
    )
    
    email_content = f"Remetente: {email.remetente}\nAssunto: {email.assunto}\nDescrição: {email.descricao}"
    ai_result = call_ai_api(email_content, nlp_data)
    quality_score = nlp_processor.get_text_quality_score(nlp_data['features'])
    
    return EmailResponse(
        classificacao=ai_result["classificacao"],
        rating=ai_result["rating"],
        motivo=ai_result["motivo"],
        resposta_sugerida=ai_result["resposta_sugerida"],
        nlp_features=nlp_data['features'],
        quality_score=quality_score
    )

def is_tesseract_available() -> bool:
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def extract_pdf_text(file_content: bytes) -> str:
    try:
        doc = fitz.open("pdf", file_content)
        texto = ""
        for page in doc:
            texto += page.get_text() + "\n"
        doc.close()
        if len(texto.strip()) > 50:
            return texto
    except Exception:
        pass
    
    try:
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            texto = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
            if len(texto.strip()) > 20:
                return texto
    except Exception:
        pass
    
    if is_tesseract_available():
        try:
            doc = fitz.open("pdf", file_content)
            texto = ""
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img_pil = Image.open(io.BytesIO(img_data))
                ocr_text = pytesseract.image_to_string(img_pil, lang='por+eng')
                if ocr_text and len(ocr_text.strip()) > 10:
                    texto += ocr_text + "\n"
            doc.close()
            if len(texto.strip()) > 30:
                return texto
        except Exception:
            pass
    
    return ""

@app.post("/classemailpdf", response_model=EmailResponse)
async def analise_email_pdf(x_api_key: str = Header(...), file: UploadFile = File(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de autenticação inválida")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são suportados")
    
    try:
        file_content = await file.read()
        texto = extract_pdf_text(file_content)
        
        if len(texto.strip()) < 5:
            raise HTTPException(status_code=400, detail="PDF não contém texto legível")
        
        nlp_data = nlp_processor.process_email_content("", "", texto)
        ai_result = call_ai_api(texto, nlp_data)
        quality_score = nlp_processor.get_text_quality_score(nlp_data['features'])
        
        return EmailResponse(
            classificacao=ai_result["classificacao"],
            rating=ai_result["rating"],
            motivo=ai_result["motivo"],
            resposta_sugerida=ai_result["resposta_sugerida"],
            nlp_features=nlp_data['features'],
            quality_score=quality_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar PDF: {str(e)}")

@app.post("/classemailtxt", response_model=EmailResponse)
async def analise_email_txt(x_api_key: str = Header(...), file: UploadFile = File(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de autenticação inválida")
    
    if not file.filename.lower().endswith('.txt'):
        raise HTTPException(status_code=400, detail="Apenas arquivos TXT são suportados")
    
    try:
        content = await file.read()
        texto = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            texto = content.decode('latin-1')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao decodificar arquivo: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo: {str(e)}")
    
    if not texto.strip():
        raise HTTPException(status_code=400, detail="Arquivo TXT está vazio")
    
    nlp_data = nlp_processor.process_email_content("", "", texto)
    ai_result = call_ai_api(texto, nlp_data)
    quality_score = nlp_processor.get_text_quality_score(nlp_data['features'])
    
    return EmailResponse(
        classificacao=ai_result["classificacao"],
        rating=ai_result["rating"],
        motivo=ai_result["motivo"],
        resposta_sugerida=ai_result["resposta_sugerida"],
        nlp_features=nlp_data['features'],
        quality_score=quality_score
    )
