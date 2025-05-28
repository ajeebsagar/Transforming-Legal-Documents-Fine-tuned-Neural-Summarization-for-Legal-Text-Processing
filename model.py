from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Union
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download required NLTK resources
def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            logging.info(f"Resource {resource} already downloaded")
        except LookupError:
            try:
                nltk.download(resource)
                logging.info(f"Successfully downloaded {resource}")
            except Exception as e:
                logging.error(f"Error downloading {resource}: {str(e)}")

# Download resources at module level
download_nltk_resources()

class LegalDocumentSummarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the legal document summarizer.
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.stop_words = set(stopwords.words('english'))
        logging.info(f"Model loaded on {self.device}")

    def _create_sentence_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """Create similarity matrix for sentences"""
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 != idx2:
                    similarity_matrix[idx1][idx2] = self._sentence_similarity(
                        sentences[idx1], 
                        sentences[idx2]
                    )
        return similarity_matrix

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences"""
        words1 = [word.lower() for word in word_tokenize(sent1) if word.lower() not in self.stop_words]
        words2 = [word.lower() for word in word_tokenize(sent2) if word.lower() not in self.stop_words]
        
        all_words = list(set(words1 + words2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for word in words1:
            vector1[all_words.index(word)] += 1
        
        for word in words2:
            vector2[all_words.index(word)] += 1
            
        return 1 - cosine_distance(vector1, vector2)

    def generate_extractive_summary(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate extractive summary using TextRank algorithm
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
        Returns:
            str: Extractive summary
        """
        try:
            # Tokenize text into sentences
            sentences = sent_tokenize(text)
            
            if len(sentences) <= num_sentences:
                return text
                
            # Create similarity matrix
            similarity_matrix = self._create_sentence_similarity_matrix(sentences)
            
            # Create graph and calculate scores
            graph = nx.from_numpy_array(similarity_matrix)
            scores = nx.pagerank(graph)
            
            # Get top sentences
            ranked_sentences = sorted(
                ((scores[i], sentence) for i, sentence in enumerate(sentences)),
                reverse=True
            )
            
            # Select top N sentences and sort by original position
            selected_indices = [sentences.index(sent) for _, sent in ranked_sentences[:num_sentences]]
            selected_indices.sort()
            
            # Combine sentences in original order
            summary = ' '.join(sentences[i] for i in selected_indices)
            
            return summary
            
        except Exception as e:
            logging.error(f"Error in extractive summarization: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess the input text by splitting it into chunks that fit the model's max length.
        Args:
            text (str): Input legal document text
        Returns:
            List[str]: List of text chunks
        """
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logging.error(f"Error in sentence tokenization: {str(e)}")
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.tokenize(sentence)
            sentence_length = len(sentence_tokens)
            
            if current_length + sentence_length <= self.tokenizer.model_max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

    def generate_summary(self, text: str, max_length: int = 150, min_length: int = 50, 
                        method: str = "abstractive", num_sentences: int = 3) -> Dict[str, Union[str, float]]:
        """
        Generate a summary for the given legal document.
        Args:
            text (str): Input legal document text
            max_length (int): Maximum length of the generated summary
            min_length (int): Minimum length of the generated summary
            method (str): Summarization method ("abstractive" or "extractive")
            num_sentences (int): Number of sentences for extractive summary
        Returns:
            Dict[str, Union[str, float]]: Dictionary containing summary and metadata
        """
        try:
            if method == "extractive":
                summary = self.generate_extractive_summary(text, num_sentences)
            else:  # abstractive
                chunks = self.preprocess_text(text)
                summaries = []
                
                for chunk in chunks:
                    inputs = self.tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
                    inputs = inputs.to(self.device)
                    
                    summary_ids = self.model.generate(
                        inputs["input_ids"],
                        num_beams=4,
                        max_length=max_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        early_stopping=True
                    )
                    
                    summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    summaries.append(summary)
                
                summary = " ".join(summaries)
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text),
                "method": method
            }
        except Exception as e:
            logging.error(f"Error in generate_summary: {str(e)}")
            raise

    def evaluate_summary(self, original_text: str, generated_summary: str) -> Dict[str, float]:
        """
        Evaluate the quality of the generated summary.
        Args:
            original_text (str): Original legal document text
            generated_summary (str): Generated summary
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        compression_ratio = len(generated_summary) / len(original_text)
        
        return {
            "compression_ratio": compression_ratio,
        } 