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


AI_BLACK_AND_WHITE_IMAGE_COLORIZATION_WITH_OPENCV/
│
├── app.py                     # Streamlit web application
├── models/
│   ├── colorization_deploy_v2.prototxt
│   ├── colorization_release_v2.caffemodel
│   └── pts_in_hull.npy
│
├── Input_images/              # Sample test images
├── Result_images/             # Generated colorized outputs
├── requirements.txt
└── README.md



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

# 📝 **Project Description (For Resume / Portfolio)**

**AI Black & White Image Colorization** is a deep learning–based web application that converts grayscale images into realistic colored images using OpenCV’s DNN framework and a pretrained Caffe neural network. The system extracts lightness information from black & white images and predicts the chrominance channels using a convolutional neural network trained on millions of images. The application is deployed using Streamlit, providing an intuitive web interface for real-time image upload and visualization. This project demonstrates practical applications of computer vision, neural networks, and image processing in restoring and enhancing visual data.


