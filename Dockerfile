FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias para Tkinter + OpenCV + soporte X11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    tk \
    tcl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el proyecto
COPY . /app

# El modelo NO se copia: se montará como volumen en /app/models
CMD ["python", "ui/main.py"]