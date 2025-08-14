from flask import Flask ,request,jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask_cors import CORS



app =Flask(__name__)
CORS(app)  # Autorise toutes les origines

@app.route('/')
def index():
    return "APIs Is working fine 🟢"


################ 1. OCR API ################
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
modelOCR = ocr_predictor(pretrained=True)

def extract_text_from_image(file_path) :
    output =""
    doc =DocumentFile.from_images(file_path)
    print("Inference starting 🟢...")
    startingTime = time.time()
    result = modelOCR(doc)
    endingTime =time.time()
    print(f"Inference End ,Duration :{endingTime-startingTime}👌...")
    for block in result.pages[0].blocks :
        for line in block.lines :
            output = output +" ".join([word.value for word in line.words]) +","
    return output

# Dossier pour stocker temporairement les fichiers
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_ocr_image',methods=['POST'])
def get_ocr():
    if "ocr_image" not in request.files :
        return jsonify({'error': 'Aucun fichier envoyé nommé image'}), 400
    image = request.files['ocr_image']
    print("1️⃣ IMGAE SAVED from upload_ocr_image API")

    filename = secure_filename(image.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    image.save(file_path) # Sauvegarde du fichier
    # Appel de ta fonction d’analyse
    print("2️⃣Start Extract Process from detect_object API ....")

    extracted_text = extract_text_from_image(file_path)
    print("3️⃣END Extract Process from detect_object API ....")

    return jsonify({'text': str(extracted_text)})









################ 2. Objet detetcion API ################
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

@app.route("/detect_object",methods=['POST'])
def detect_object():
    if "obj_image" not in request.files :
        return jsonify({'error': 'Aucun fichier envoyé nommé image'}), 400
    image = request.files['obj_image']

    filename = secure_filename(image.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    # 🟩 Sauvegarde du fichier dans le dossier
    image.save(file_path)

    all_classes =set()
    results = model(file_path)
    for result in results :
        boxes = result.boxes 
        for box in boxes :
            classId = box.cls
            className = model.names[int(classId)]
            all_classes.add(className)
    all_classes = list(all_classes)
    result = " , ".join(all_classes)

    output = {
        "classes" : result
    }
    extra =request.args.get('extra')
    if extra : 
        output["extra"]= extra


    return jsonify(output),200









################ 3. Similarity Match ################

from sentence_transformers import SentenceTransformer, util
import time 

model =SentenceTransformer('all-MiniLM-L6-v2')
@app.route("/match",methods=['POST'])
def match():

    data=request.get_json()
    text1 ,text2 =  data["ocr_text"],data["detected_obj"]
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)

    startTime = time.time()
    score3 = (util.pytorch_cos_sim(emb1, emb2)[0][0]*100).item()
    endTime= time.time()

    print("End Trransformer test " , endTime-startTime)

    return jsonify({"score":round(score3,2)}),200

################ 2. Objet detetcion API ################


