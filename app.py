import os
from flask import Flask, render_template, request, jsonify
import torch
from PIL import Image
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from colorthief import ColorThief
import groq
import base64
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
groq_api_key = "YOUR_API_KEY"
client = groq.Groq(api_key=groq_api_key)
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_face(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if model.config.id2label[label.item()] == "person":
            box = [round(i, 2) for i in box.tolist()]
            return image.crop(box)
   
    return None

def extract_face_color(face_image):
    temp_path = "temp_face.jpg"
    face_image.save(temp_path)
    color_thief = ColorThief(temp_path)
    dominant_color = color_thief.get_color(quality=1)
    os.remove(temp_path)
   
    return dominant_color

def get_llm_recommendations(face_color):
    r, g, b = face_color
    prompt = f"""Based on modern fashion data, recommend three dresses for someone with a facial complexion color of RGB({r}, {g}, {b}). Consider modern color pairings and cultural preferences. Format your response as a list of three dress recommendations, each on a new line starting with a dash (-). Include the suggested color of each dress.Dont mention the face complex and explain how well it can suit to the person in exciting way"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a fashion expert with knowledge of modern color preferences in clothing and color theory.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-70b-versatile",
        max_tokens=200,
    )

    return chat_completion.choices[0].message.content.strip().split('\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'capture.jpg')
    image_file.save(filename)
   
    image = Image.open(filename)

    face = detect_face(image)
    if face is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    face_color = extract_face_color(face)
    recommendations = get_llm_recommendations(face_color)

    return jsonify({
        'face_color': face_color,
        'recommendations': recommendations,
        'image_path': f'/{filename}'
    })

if __name__ == '__main__':
    app.run(debug=True)
