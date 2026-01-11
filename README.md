# 🎨 AI Black & White Image Colorization using OpenCV & Deep Learning

This project is an AI-powered web application that converts black & white images into realistic color images using a deep learning model and OpenCV’s DNN module. The system uses a pretrained Caffe model trained on millions of natural images to predict color information from grayscale input.

The application is built using **Python**, **OpenCV**, and **Streamlit**, providing an easy-to-use web interface for uploading images and instantly viewing colorized results.

---

## 🚀 Features

- Upload any black & white image
- AI-based automatic colorization
- Deep learning powered by Caffe model
- Real-time processing
- Web-based interface using Streamlit
- Supports JPG and PNG images
- Download-ready output

---

## 🧠 Technology Stack

| Component | Technology |
|--------|-----------|
| Programming Language | Python |
| Frontend | Streamlit |
| Image Processing | OpenCV |
| Deep Learning Model | Caffe (Colorization model) |
| Numerical Computing | NumPy |
| Image Handling | Pillow |

---

## 🏗 Project Architecture


AI_BLACK_AND_WHITE_IMAGE_COLORIZATION_WITH_OPENCV/ <br>
│<br>
├─ app.py                     # Streamlit web application<br>
├─ models/<br>
│   ├── colorization_deploy_v2.prototxt<br>
│   ├── colorization_release_v2.caffemodel<br>
│   └── pts_in_hull.npy<br>
│<br>
├── Input_images/              # Sample test images<br>
├── Result_images/             # Generated colorized outputs<br>
├── requirements.txt<br>
└── README.md<br>



---

## ⚙ How It Works

1. The uploaded image is converted to grayscale.
2. The grayscale image is converted to LAB color space.
3. The L channel (lightness) is passed into a deep learning neural network.
4. The model predicts the a and b color channels.
5. The predicted channels are combined with the original L channel.
6. The LAB image is converted back to RGB, producing a realistic color image.

---

## 🧪 AI Model

The project uses a pretrained **Caffe colorization model** trained on ImageNet, capable of predicting realistic colors from black & white images.

Model files:
- `colorization_deploy_v2.prototxt`
- `colorization_release_v2.caffemodel`
- `pts_in_hull.npy`

These files store neural network architecture, learned weights, and color cluster centers.

---

## 💻 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-image-colorizer.git
cd ai-image-colorizer
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the application
```bash
streamlit run app.py
```

Open browser:
```bash
http://localhost:8501
```

👨‍💻 Author

Rahul Roy
Full-Stack Developer & AI Enthusiast
India 🇮🇳


---



