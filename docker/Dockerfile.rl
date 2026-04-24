FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libgomp1 \
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch-geometric==2.5.2

COPY src/ /workspace/src/
COPY notebooks/ /workspace/notebooks/
COPY data/ /workspace/data/
COPY checkpoints/ /workspace/checkpoints/

ENV PYTHONPATH=/workspace
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", \
     "notebooks/03_rl_generator.ipynb", \
     "--ExecutePreprocessor.timeout=7200"]
