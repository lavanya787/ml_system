import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ANALYSIS_FOLDER = os.path.join(BASE_DIR, 'analysis')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
PREVIEW_FOLDER = os.path.join(BASE_DIR, 'previews')
LOGS_FOLDER = os.path.join(BASE_DIR, 'logs')

ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

# Create folders if not exist
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, PREVIEW_FOLDER, LOGS_FOLDER, ANALYSIS_FOLDER]:
    os.makedirs(folder, exist_ok=True)
