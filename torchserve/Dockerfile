FROM pytorch/torchserve:latest-cpu

USER root
RUN /home/venv/bin/pip install --no-cache-dir sentence-transformers langchain-community faiss-cpu

COPY model-store/mailru_rag.mar /home/model-server/model-store/mailru_rag.mar

EXPOSE 8080 8081 8082

CMD ["torchserve","--start","--ncs","--disable-token-auth","--model-store","/home/model-server/model-store","--models","mailru_rag=mailru_rag.mar","--foreground"]
