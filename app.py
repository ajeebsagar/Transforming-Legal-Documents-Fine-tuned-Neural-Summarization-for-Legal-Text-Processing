import streamlit as st
import torch
from model import LegalDocumentSummarizer
import logging
import warnings
import os
import io


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing optional dependencies with explicit logging
try:
    import PyPDF2
    logger.info("Successfully loaded PyPDF2")
    PDF_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PyPDF2 import failed: {str(e)}")
    PDF_AVAILABLE = False

try:
    import chardet
    logger.info("Successfully loaded chardet")
    CHARDET_AVAILABLE = True
except ImportError as e:
    logger.warning(f"chardet import failed: {str(e)}")
    CHARDET_AVAILABLE = False

try:
    import docx2txt
    logger.info("Successfully loaded docx2txt")
    DOCX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"docx2txt import failed: {str(e)}")
    DOCX_AVAILABLE = False

def read_file_content(uploaded_file):
    """Read content from different file types with encoding detection"""
    try:
        # Get file extension
        file_type = uploaded_file.name.split('.')[-1].lower()
        logger.info(f"Processing file of type: {file_type}")
        
        if file_type == 'txt':
            # Read the file content as bytes first
            content_bytes = uploaded_file.read()
            
            if CHARDET_AVAILABLE:
                # Detect the encoding
                result = chardet.detect(content_bytes)
                encoding = result['encoding'] if result['encoding'] else 'utf-8'
                logger.info(f"Detected encoding: {encoding}")
            else:
                # If chardet is not available, try common encodings
                encoding = 'utf-8'
                logger.info("Using default utf-8 encoding")
            
            try:
                # Try detected encoding
                return content_bytes.decode(encoding)
            except UnicodeDecodeError:
                # Fallback encodings
                for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        return content_bytes.decode(enc)
                    except UnicodeDecodeError:
                        continue
                raise UnicodeDecodeError(f"Could not decode file with any common encoding")
                
        elif file_type == 'docx':
            if not DOCX_AVAILABLE:
                raise ImportError("docx2txt package is not installed. Please install it to read .docx files.")
            # Extract text from docx
            text = docx2txt.process(uploaded_file)
            return text
            
        elif file_type == 'pdf':
            if not PDF_AVAILABLE:
                raise ImportError("PyPDF2 package is not installed. Please install it to read .pdf files.")
            # Read PDF file
            try:
                pdf_bytes = io.BytesIO(uploaded_file.getvalue())
                pdf_reader = PyPDF2.PdfReader(pdf_bytes)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                logger.info(f"Successfully extracted text from PDF with {len(pdf_reader.pages)} pages")
                return text
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                raise
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

# Set page config
st.set_page_config(
    page_title="Legal Document Summarizer",
    page_icon="⚖️",
    layout="wide"
)

@st.cache_resource(show_spinner=True)
def load_model():
    """Load the model and return an instance of LegalDocumentSummarizer"""
    try:
        # Check if fine-tuned model exists
        model_path = "fine_tuned_model/final_model"
        if os.path.exists(model_path):
            return LegalDocumentSummarizer(model_name=model_path)
        else:
            return LegalDocumentSummarizer()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logging.error(f"Model loading error: {str(e)}")
        return None

def main():
    st.title("Legal Document Summarizer")
    st.write("Upload or paste your legal document to generate a summary.")

    # Initialize model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please try again.")
        return

    # Show supported file types based on available packages
    supported_types = ["txt"]
    if DOCX_AVAILABLE:
        supported_types.append("docx")
    if PDF_AVAILABLE:
        supported_types.append("pdf")
    
    logger.info(f"Available file types: {supported_types}")

    # Input methods
    input_method = st.radio("Choose input method:", ["Upload File", "Paste Text"])

    input_text = ""
    if input_method == "Upload File":
        st.info(f"Supported file types: {', '.join(supported_types)}")
        uploaded_file = st.file_uploader("Choose a file", type=supported_types)
        if uploaded_file:
            try:
                with st.spinner(f"Reading {uploaded_file.name}..."):
                    input_text = read_file_content(uploaded_file)
                st.success(f"Successfully read file: {uploaded_file.name}")
            except ImportError as e:
                st.error(str(e))
                st.info("Please install the required package or use the paste text option.")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                st.info("Try uploading a different file or use the paste text option.")
    else:
        input_text = st.text_area("Paste your legal document here:", height=300)

    # Summarization parameters
    st.subheader("Summarization Settings")
    
    # Method selection
    method = st.radio(
        "Choose summarization method:",
        ["Abstractive", "Extractive"],
        help="Abstractive: Generates new text. Extractive: Selects important sentences from the original text."
    )
    
    col1, col2 = st.columns(2)
    
    if method == "Abstractive":
        with col1:
            max_length = st.slider("Maximum summary length", 50, 500, 150)
        with col2:
            min_length = st.slider("Minimum summary length", 30, 200, 50)
    else:  # Extractive
        with col1:
            num_sentences = st.slider("Number of sentences to extract", 3, 10, 5)
            max_length = 500  # Default for extractive
            min_length = 50   # Default for extractive

    if st.button("Generate Summary") and input_text:
        if len(input_text.strip()) < 100:
            st.warning("The input text is too short. Please provide a longer document.")
            return
            
        with st.spinner("Generating summary..."):
            try:
                result = model.generate_summary(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    method=method.lower(),
                    num_sentences=num_sentences if method == "Extractive" else 3
                )
                
                st.subheader("Generated Summary")
                st.write(result["summary"])
                
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Length", f"{result['original_length']} chars")
                with col2:
                    st.metric("Summary Length", f"{result['summary_length']} chars")
                with col3:
                    st.metric("Compression Ratio", f"{result['compression_ratio']:.2%}")
                with col4:
                    st.metric("Method", result["method"].title())
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")
                logging.error(f"Summary generation error: {str(e)}")
    
    st.markdown("---")
    st.markdown("""
    ### Tips
    - For best results, ensure your document is well-formatted
    - The model works best with documents between 1,000 and 15,000 characters
    - Abstractive summarization generates new text, while extractive selects important sentences
    - Extractive summarization is faster but less flexible
    - Currently supported file formats: {', '.join(supported_types)}
    """)

if __name__ == "__main__":
    main() 