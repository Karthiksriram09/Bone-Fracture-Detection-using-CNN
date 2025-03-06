# 🦴 Bone Fracture Detection using CNN  

## 📌 Overview  
This project implements a **Convolutional Neural Network (CNN)** for detecting bone fractures from **X-ray images**. The model classifies images as **fractured or non-fractured**, assisting in **automated medical diagnosis** and reducing manual analysis time.  

## 🚀 Features  
✅ Automated bone fracture detection from X-ray images  
✅ Preprocessing techniques for enhanced image quality  
✅ Deep learning model trained using CNN for high accuracy  
✅ Visualization of predictions using Grad-CAM  
✅ User-friendly interface for easy image upload and diagnosis  

## 🛠 Tech Stack  
- **Python**  
- **TensorFlow/Keras** (for CNN model)  
- **OpenCV** (for image processing)  
- **Matplotlib & Seaborn** (for visualization)  
- **Flask or Streamlit** (if deploying with a UI)  

## 📂 Dataset  
The model is trained on a dataset of X-ray images labeled as **fractured** and **non-fractured**. You can use publicly available datasets like **MURA (Musculoskeletal Radiographs)** or any other medical dataset.  

## 🔧 Installation & Setup  
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/YourUsername/Bone-Fracture-Detection.git  
   cd Bone-Fracture-Detection
   ```
2. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the model training script:**  
   ```bash
   python train.py
   ```
4. **For testing the model:**  
   ```bash
   python test.py --image path/to/image.jpg
   ```
5. **Deploy (if applicable):**  
   ```bash
   streamlit run app.py
   ```

## 🖼 Model Architecture  
The CNN model consists of:  
- **Convolutional Layers** (feature extraction)  
- **Max Pooling Layers** (dimensionality reduction)  
- **Fully Connected Layers** (classification)  
- **Softmax Activation** (binary classification)  

## 📊 Results  
- Achieved **high accuracy** in bone fracture detection  
- Reduced false positives with **data augmentation** and **hyperparameter tuning**  
- Improved interpretability using **Grad-CAM visualization**  

## 🏥 Applications  
- **Medical Diagnosis**: Assists radiologists in detecting fractures quickly  
- **AI in Healthcare**: Enhances efficiency and reduces human error  
- **Remote Diagnosis**: Can be deployed for telemedicine services  

## 🤝 Contribution  
Feel free to fork this repository and contribute by improving the model, adding new features, or optimizing performance!  
