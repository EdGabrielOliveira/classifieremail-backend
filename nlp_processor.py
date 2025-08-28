import re
import nltk
from typing import List, Dict
import unicodedata

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('stemmers/rslp')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('rslp')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

class EmailNLPProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('portuguese'))
        self.stemmer = RSLPStemmer()

        self.stop_words.update([
            'email', 'mensagem', 'assunto', 'remetente', 'destinatario',
            'para', 'de', 'em', 'por', 'com', 'sem', 'sobre', 'ate',
            'desde', 'durante', 'antes', 'depois', 'entre', 'contra',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through'
        ])
    
    def clean_text(self, text: str) -> str:
        
        if not text:
            return ""
        
        text = unicodedata.normalize('NFKD', text)

        text = text.lower()

        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        text = re.sub(r'\S+@\S+', '', text)

        text = re.sub(r'[\+]?[1-9]?[0-9]{7,15}', '', text)

        text = re.sub(r'[^a-záàâãéêíóôõúç\s]', ' ', text)

        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_and_filter(self, text: str) -> List[str]:

        tokens = word_tokenize(text, language='portuguese')

        filtered_tokens = [
            token for token in tokens 
            if token.lower() not in self.stop_words 
            and len(token) > 2
            and token.isalpha()
        ]
        
        return filtered_tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]
    
    def extract_features(self, text: str) -> Dict:
        if not text:
            return {
                "word_count": 0,
                "sentence_count": 0,
                "avg_word_length": 0,
                "question_marks": 0,
                "exclamation_marks": 0,
                "uppercase_ratio": 0,
                "spelling_errors": 0
            }
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "question_marks": text.count('?'),
            "exclamation_marks": text.count('!'),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
        }

        error_indicators = [
            word for word in words 
            if len(word) > 20 or (len(word) == 1 and word.lower() not in ['a', 'e', 'i', 'o', 'u'])
        ]
        features["spelling_errors"] = len(error_indicators)
        
        return features
    
    def process_email_content(self, remetente: str, assunto: str, descricao: str) -> Dict:

        full_text = f"{remetente} {assunto} {descricao}"

        cleaned_text = self.clean_text(full_text)

        tokens = self.tokenize_and_filter(cleaned_text)

        stemmed_tokens = self.stem_tokens(tokens)

        features = self.extract_features(full_text)

        processed_data = {
            "original": {
                "remetente": remetente,
                "assunto": assunto,
                "descricao": descricao
            },
            "cleaned": {
                "full_text": cleaned_text,
                "tokens": tokens,
                "stemmed_tokens": stemmed_tokens
            },
            "features": features,
            "processed_text": " ".join(stemmed_tokens)
        }
        
        return processed_data
    
    def get_text_quality_score(self, features: Dict) -> float:
        score = 5.0

        if features["word_count"] < 3:
            score -= 2
        elif features["word_count"] > 200:
            score -= 1

        if features["spelling_errors"] > 2:
            score -= features["spelling_errors"] * 0.5

        if features["uppercase_ratio"] > 0.7:
            score -= 2

        if features["sentence_count"] > 1 and features["avg_word_length"] > 3:
            score += 1
        
        return max(0, min(10, score))
