# Imagem de execução da solução (regras + GLiNER2.5 fine-tuned).
#   docker build -t cacador .
#   docker run --rm --gpus all --network none -v /dados:/dados cacador \
#       /dados/base.db /dados/txt /dados/submission.csv
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pesos em revisão fixa, conferidos por SHA-256; a execução não acessa a internet
COPY baixar_modelo.py .
RUN python baixar_modelo.py modelos/gliner25_ft5

COPY cacador/ cacador/
COPY run_baseline.py run.sh CONFIANCA.json ./

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
ENTRYPOINT ["bash", "/app/run.sh"]
