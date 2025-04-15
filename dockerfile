# Utilise Python 3.10 avec Debian slim
FROM python:3.10-slim

# Empêche les messages interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Installe les dépendances système nécessaires à OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crée le dossier de travail
WORKDIR /app

# Copie les fichiers du projet dans l'image
COPY . .

# Installe les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port utilisé par Flask (si nécessaire)
EXPOSE 8080

# Démarre l'application Flask
CMD ["python", "app.py"]
