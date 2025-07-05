# 🧠 Face Recognition & Gender Classification – FACECOM Challenge 2025

This project is submitted as part of the **FACECOM Hackathon 2025**, focusing on building robust facial analysis models in visually challenging environments (blur, rain, fog, glare, etc.).

---

## 📝 Problem Statement

Face recognition algorithms typically suffer performance drops under non-ideal conditions. This challenge involves developing:

- **Task A:** Gender Classification (Male/Female)
- **Task B:** Face Recognition (Person Identity)

Using the **FACECOM dataset**, which contains 5000+ face images under real-world challenging scenarios.

---

## 📁 Dataset Details

**FACECOM Dataset** contains:
- 🔹 Images with visual degradations (blur, fog, rain, low light)
- 🔹 Binary gender labels (Male / Female)
- 🔹 Multi-class identity labels for face recognition

### 📊 Dataset Split
- **Training**: 70%
- **Validation**: 15%
- **Test (Hidden)**: 15%

---

## 🚀 Tasks Overview

### ✅ Task A – Gender Classification
- Binary classification of face images
- Model: `Your model name (e.g., ResNet18)`
- Metrics: Accuracy, Precision, Recall, F1-score

### ✅ Task B – Face Recognition
- Multi-class classification for identifying individuals
- Model: `Your model name (e.g., Custom CNN / EfficientNet)`
- Metrics: Top-1 Accuracy, Macro-averaged F1-score

---

## 🧠 Our Approach

- Applied data augmentation to handle visual distortions
- Used transfer learning and pretrained CNN architectures
- Integrated Batch Normalization, Dropout for generalization
- Trained using Adam optimizer with learning rate scheduling

---

## 🧪 Evaluation Results (on Validation Set)

| Task               | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Gender Classification | XX%      | XX%       | XX%    | XX%      |
| Face Recognition (Top-1) | XX%  | -         | -      | XX%      |

> Replace XX% with your actual results

---

]
   Team Details
   soumyodip thanadar  (team leader)
   

[Teammate 2 name]
Arpan maity

[Teammate 3 name]
prasanta adak

💡 Innovations / Highlights
Trained under low-quality image scenarios

Achieved good generalization with minimal overfitting

Modular code with clear documentation

Optimized for both classification and recognition tasks
