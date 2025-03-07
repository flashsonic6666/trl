# Updated app.py with SMILES validation

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import tiktoken
import base64
from PIL import Image
import io
import re
from rdkit import Chem

app = Flask(__name__)

# Configuration - update these paths
DATA_JSON_PATH = "file_to_smiles.json"  # JSON mapping filenames to SMILES strings
ANNOTATIONS_PATH = "dumb_annotations.json"  # Path to save annotations
IMAGES_DIR = "/data/rbg/users/richwang/trl/indigo_simple_render/images"  # Directory containing the images

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load data
def load_data():
    with open(DATA_JSON_PATH, 'r') as f:
        data = json.load(f)
    return data

# Load annotations
def load_annotations():
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, 'r') as f:
            return json.load(f)
    return {}

# Save annotations
def save_annotations(annotations):
    with open(ANNOTATIONS_PATH, 'w') as f:
        json.dump(annotations, f, indent=2)

# Get image as base64
def get_image_base64(filename):
    try:
        img_path = os.path.join(IMAGES_DIR, filename)
        img = Image.open(img_path)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        return None

# Canonicalize SMILES
def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol)
        return None
    except:
        return None

# Extract answer from annotation
def extract_answer(annotation):
    match = re.search(r'<answer>(.*?)</answer>', annotation)
    if match:
        return match.group(1).strip()
    return None

# Check if the annotation contains the correct answer
def check_answer(ground_truth, annotation):
    answer = extract_answer(annotation)
    if not answer:
        return False
    
    canonical_ground_truth = canonicalize_smiles(ground_truth)
    canonical_answer = canonicalize_smiles(answer)
    
    if not canonical_ground_truth or not canonical_answer:
        return False
    
    return canonical_ground_truth == canonical_answer

# Count correct answers
def count_correct_answers():
    data = load_data()
    annotations = load_annotations()
    
    correct_count = 0
    total_count = 0
    
    for filename, ground_truth in data.items():
        if filename in annotations:
            annotation = annotations[filename]
            if check_answer(ground_truth, annotation):
                correct_count += 1
            total_count += 1
    
    return {
        "correct": correct_count,
        "total": total_count
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    data = load_data()
    annotations = load_annotations()
    filenames = list(data.keys())
    total_images = len(filenames)
    
    correct_stats = count_correct_answers()
    
    return jsonify({
        "total_images": total_images,
        "filenames": filenames,
        "correct_count": correct_stats["correct"],
        "total_annotated": correct_stats["total"]
    })

@app.route('/api/image/<filename>')
def get_image_data(filename):
    data = load_data()
    annotations = load_annotations()
    
    image_base64 = get_image_base64(filename)
    ground_truth = data.get(filename, "")
    annotation = annotations.get(filename, "")
    token_count = len(tokenizer.encode(annotation))
    
    is_correct = check_answer(ground_truth, annotation)
    
    return jsonify({
        "filename": filename,
        "image": image_base64,
        "ground_truth": ground_truth,
        "annotation": annotation,
        "token_count": token_count,
        "is_correct": is_correct
    })

@app.route('/api/save', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data.get('filename')
    annotation = data.get('annotation')
    
    ground_truth = load_data().get(filename, "")
    is_correct = check_answer(ground_truth, annotation)
    
    annotations = load_annotations()
    annotations[filename] = annotation
    save_annotations(annotations)
    
    token_count = len(tokenizer.encode(annotation))
    
    # Update correct answers count
    correct_stats = count_correct_answers()
    
    return jsonify({
        "success": True,
        "filename": filename,
        "token_count": token_count,
        "is_correct": is_correct,
        "correct_count": correct_stats["correct"],
        "total_annotated": correct_stats["total"]
    })

@app.route('/api/count_tokens', methods=['POST'])
def count_tokens():
    data = request.json
    text = data.get('text', '')
    token_count = len(tokenizer.encode(text))
    
    # If we have a filename, check if the annotation is correct
    filename = data.get('filename')
    is_correct = False
    
    if filename:
        ground_truth = load_data().get(filename, "")
        is_correct = check_answer(ground_truth, text)
    
    return jsonify({
        "token_count": token_count,
        "is_correct": is_correct
    })

if __name__ == '__main__':
    # Create directories if they don't exist
    #os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)