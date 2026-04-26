FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libgomp1 \
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir torch-geometric==2.5.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev curl gnupg \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor \
       -o /usr/share/keyrings/microsoft-prod.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
       https://packages.microsoft.com/debian/12/prod bookworm main" \
       > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ /workspace/src/
COPY notebooks/ /workspace/notebooks/
RUN mkdir -p /workspace/data /workspace/checkpoints

ENV PYTHONPATH=/workspace
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", \
     "notebooks/03_rl_generator.ipynb", \
     "--ExecutePreprocessor.timeout=7200"]
