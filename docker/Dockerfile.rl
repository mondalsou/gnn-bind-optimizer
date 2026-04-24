FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.2.0+cpu.html

COPY src/ /workspace/src/
COPY notebooks/ /workspace/notebooks/
COPY data/ /workspace/data/
COPY checkpoints/ /workspace/checkpoints/

ENV PYTHONPATH=/workspace
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", \
     "notebooks/phase3_rl_generator.ipynb", \
     "--ExecutePreprocessor.timeout=3600"]
