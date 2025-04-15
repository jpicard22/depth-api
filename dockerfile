FROM python:3.10-slim

# Installation des dépendances système (dont libGL)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de l'application
WORKDIR /app
COPY . /app

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Lancement de l'app Flask
CMD ["python", "app.py"]
