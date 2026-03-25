# News Research Tool 🚀

## Live Demo
App running at [http://localhost:8501](http://localhost:8501)

## Quick Start (Local)
```bat
.\run_app.bat
```

## Deployment Options

### 1. **Streamlit Cloud** (Fixed ✅)
```
1. Rename requirements.txt → requirements_old.txt
2. Rename requirements_cloud.txt → requirements.txt  
3. git add . && git commit -m "Fix deps for cloud"
4. Push to GitHub
5. streamlit.io/cloud → Deploy → Select branch/main
```
**Fixed Issues**:
- `faiss-cpu==1.7.4` → `>=1.12.0` (compatible)
- `langchain>=0.1.0` (no deprecation)
- Added `torch/transformers` explicit

### 2. **Render.com** (Free)
```
1. requirements.txt (already exists)
2. render.com → New → Web Service → GitHub repo
3. Build: `pip install -r requirements.txt`
4. Start: `streamlit run app.py --server.port $PORT`
```

### 3. **Docker** (Production)
```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Dependencies Note
```
⚠️ langchain 0.0.353 works but upgrade recommended:
pip install langchain>=0.1.0 langchain-community>=0.1.0
```

## Features Verified ✅
- Stock target extraction (₹100-5000)
- Buy/Sell recommendations  
- General Q&A with RAG
- Summarization
- Token-safe T5-large LLM

**Deploy now → Live stock research tool for anyone!**
