
# 🧠 3D Image Reconstruction from 2D Views using Vision Transformer (ViT)

A full-stack deep learning web application that reconstructs 3D objects from single 2D views using a Vision Transformer (ViT)-based model. This project showcases a seamless integration of cutting-edge deep learning, interactive UI, and real-time 3D visualization.

## 🔍 Overview

This system takes 2D images from single viewpoints as input, processes them through a transformer-based deep learning model, and outputs a 3D model which can be visualized and downloaded in `.OBJ` format.

## 🚀 Tech Stack

- **Frontend**: React.js  
- **Backend**: Python Flask (REST API)  
- **Model**: Vision Transformer (ViT) for multi-view 3D reconstruction  
- **3D Visualization**: Three.js or similar WebGL library  

## 🔧 Key Features

- 🎯 **Transformer-based 3D Reconstruction**: Uses ViT with self-attention for accurate feature extraction and integration across views.  
- 🖼️ **Multi-View Input**: Upload multiple 2D images captured from various angles.  
- 🧠 **Deep Learning Inference API**: Flask backend performs inference and generates the 3D model.  
- 🌐 **Interactive Web Interface**: Intuitive React-based frontend for uploading images, visualizing the output, and downloading `.OBJ` files.  
- 🧊 **3D Viewer Integration**: Embedded real-time 3D preview of reconstructed objects using Three.js.

## 📦 Project Structure

```bash
├── static/                   # Syle and Javascript flies
│   ├── css/
        └── style.css
│   └── js/
│       └── main.js
├── templates/                # HTML template file
│   └── Index.html
├── app.py                    # Flask API
├── generation_code.ipynb     # Model Generation Code
└── requirements.txt
```

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/RatneshKJaiswal/Imagin3D
cd Imagin3D
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
python app.py
```

## 🖼️ Example Usage

1. Upload multiple 2D views of an object
2. Preview the reconstructed 3D model on the webpage usind Three.js
3. Download the model as an `.OBJ` file

## 🧠 Model Details

The ViT-based model leverages self-attention to learn spatial dependencies across image views, enabling robust feature fusion and accurate 3D reconstruction.

## 📄 License

This project is licensed under the MIT License.

## 🤝 Acknowledgments

- Vision Transformer (ViT) research community  
- Javascript and Flask documentation  
- Three.js for 3D visualization

## 🎓 Representation Diagrams :-

### Fig.1. Web Page Sample and Preview Video

![Screenshot 2025-05-16 093548](https://github.com/user-attachments/assets/173871e6-645d-4be5-888c-bb6258a4785a)

https://github.com/user-attachments/assets/19101d34-36b7-4804-a3fa-0c5e238e7e18

### Fig.2. Training loss and accuracy graph

![Screenshot 2025-04-22 233604](https://github.com/user-attachments/assets/21db74fe-fd38-457a-b5e9-b6c5aeec5b93)

### Fig.3. Project Management Timeline

![Screenshot 2025-05-16 010930](https://github.com/user-attachments/assets/bcd86ee3-6a8b-4f28-8474-ab16ce3bbffc)

### Fig.4. ER Diagram

![Screenshot 2025-05-16 014517](https://github.com/user-attachments/assets/22d9be32-c3b6-442d-91bd-de5ed4455b5d)

### Fig.5. Data Flow Diagram

![Screenshot 2025-05-16 012939](https://github.com/user-attachments/assets/5548f6aa-34a5-4d7e-b673-fc5cb55559a7)
![Screenshot 2025-05-16 013005](https://github.com/user-attachments/assets/201e692e-cd37-44d9-827e-bb2c89eb6747)
![Screenshot 2025-05-16 013029](https://github.com/user-attachments/assets/10d560a8-5456-4d0d-843a-4b091804864c)
