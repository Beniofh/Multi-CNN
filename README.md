# Automated Recognition of Pollen Grains

This repository contains code, datasets, and instructions associated with the article:

**Bourel B., Marchant R., de Garidel-Thoron T., Tetard M., Barboni D., Gally Y., Beaufort L. (2020).**
*Automated recognition by multiple convolutional neural networks of modern, fossil, intact and damaged pollen grains.* Computers & Geoscience (in submission).

---
## 🔍 Overview

This project provides scripts to classify pollen grains using multiple Convolutional Neural Networks (CNNs). The models are trained and evaluated on datasets of **intact**, **damaged**, and **fossil** pollen grains. Single- and multi-CNN architectures are included, with and without data augmentation.

---
## 📁 Repository Structure

```
├── CNN_test_v2.1.0.py                      # Single CNN with augmentation
├── CNN_test_v2.1.0(without_augm).py       # Single CNN without augmentation
├── Multi-CNN_test_V3.5.2.py               # Multi-CNN with augmentation
├── Multi-CNN_test_V3.5.2(without_augm).py # Multi-CNN without augmentation
├── CNN_solo/                              # Individual CNN model parameters & training logs
├── intact/                                # Intact pollen dataset
├── damaged/                               # Damaged pollen dataset
├── fossil/                                # Fossil pollen dataset
└── User_manual.pdf                        # Detailed usage instructions
```

> ⚠️ Important: The folders `fossil/al`, `fossil/ca`, and `fossil/co` must remain empty. Dummy `.txt` files may need to be deleted if downloaded directly from GitHub.

---
## 🛠️ Installation

These scripts were developed and tested using:
- **Windows 8**, compatible with Linux & macOS
- **Anaconda 3 v5.2.0**
- **Python 3.6**
- **Spyder v3.3.6**
- **TensorFlow v1.14.0**

### Create Environment
1️⃣ Open *Anaconda Prompt* and create an environment:
```
conda create -n tf-3.6 python=3.6
conda activate tf-3.6
```
2️⃣ Install required dependencies from Anaconda Navigator or using:
```
conda install numpy==1.16.5 matplotlib==2.2.2 scikit-image==0.15.0 scikit-learn==0.21.2 keras==2.2.4 pillow glob2 mpmath tensorflow==1.14.0
```
3️⃣ Launch Spyder and load one of the scripts.

> ✅ All required software and libraries are open‑source.

---
## 🚀 Usage

In each Python script, choose the dataset to classify by editing:
- `switch1` → in **Multi-CNN** scripts
- `switch2` → in **Single-CNN** scripts

Datasets available:
- `"intact"`
- `"damaged"`
- `"fossil"`

Run the script in Spyder to start inference and evaluation.

---
## ✨ Scripts Summary

| Script | Model | Augmentation | Use Case |
|--------|-------|--------------|----------|
| `CNN_test_v2.1.0.py` | Single CNN | ✅ | Baseline comparison |
| `CNN_test_v2.1.0(without_augm).py` | Single CNN | ❌ | No‑augmentation testing |
| `Multi-CNN_test_V3.5.2.py` | Multi‑CNN | ✅ | Main trained model |
| `Multi-CNN_test_V3.5.2(without_augm).py` | Multi‑CNN | ❌ | Architecture comparison |

Outputs include:
- Training accuracy/loss plots (`.bmp`)
- Training logs (`.csv`)
- Trained model checkpoints (`.ckpt`)

---
## ✍️ Authors

- Benjamin Bourel
- Ross Marchant
- Thibault de Garidel‑Thoron
- Martin Tetard
- Doris Barboni
- Yves Gally
- Luc Beaufort

CEREGE, Aix‑Marseille Université, CNRS, IRD, INRA, Coll. France

📧 Contact developer: **chebenjamin@laposte.net**

---
## 📌 Citation

If you use these scripts or datasets in your research, please cite the associated article *(in submission)*.

---
## 📄 License

Open‑source — see license terms in repository (if available).

---
✅ Everything is ready to reproduce the results or extend the research!

