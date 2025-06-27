# 🚗 Driver Distraction Detection System (Real-Time AI-Powered Safety Tool)

A real-time AI-powered system built by **Utkarsh Pandey** to identify and classify distracted driver behaviors using computer vision and deep learning. This project aims to enhance road safety by actively monitoring driver activity and warning against risky behavior.

---

## 🧠 What It Does

This system uses a pre-trained Convolutional Neural Network (CNN) model (based on ResNet50) to detect **10 different types of distracted driving behaviors** using live webcam feed. It runs a Flask-based web application with a real-time video interface and delivers instant predictions.

---

## 📸 Distraction Classes Detected

- `c0`: Safe driving
- `c1`: Texting - right hand
- `c2`: Talking on phone - right hand
- `c3`: Texting - left hand
- `c4`: Talking on phone - left hand
- `c5`: Operating the radio
- `c6`: Drinking
- `c7`: Reaching behind
- `c8`: Hair or makeup
- `c9`: Talking to a passenger

---

## 🧰 Tech Stack

| Layer        | Technologies Used                              |
|--------------|-------------------------------------------------|
| Frontend     | HTML, CSS, Bootstrap (via Flask templates)      |
| Backend      | Python, Flask                                   |
| CV/AI        | OpenCV, TensorFlow / Keras, ResNet50            |
| Deployment   | Local server (Flask), can be extended to cloud  |

---

## 🚀 How to Run Locally

### Step 1: Clone this Repository
```bash
git clone https://github.com/utkarsh081/Driver-Distraction-Detection-System.git
cd Driver-Distraction-Detection-System
```

---

## 👤 Developed By :
- `Utkarsh Pandey`
- `GitHub`: https://github.com/utkarsh081
- `Email`: utkarshpandey0889@gmail.com



