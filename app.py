from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import transforms, models

app = Flask(__name__)

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('blood_model.pth', map_location=torch.device('cpu')))
model.eval()

labels = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return labels[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    label = predict(image)
    return render_template('index.html', prediction=label)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # default to 5000
    app.run(debug=True, host='0.0.0.0', port=port)  # ← KEY FIX
