FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY mp_rag_bot.py retrieval.py ./
COPY knowledge/ ./knowledge/
EXPOSE 18790
CMD ["python", "mp_rag_bot.py"]
