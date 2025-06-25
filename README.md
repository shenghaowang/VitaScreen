# VitaScreen

---

## 🚀 Features

- ✅ Diabetic risk prediction with tree based model and CNN
- ✅ Transforming tabular data into 2D image data using IGTD and NCTD algorithms
- ✅ Reproducible experiment on the CDC benchmark datasets
- ✅ Early stopping, checkpointing, and runtime logging

---

## 📦 Installation

While Docker is the recommended method for setting up the environment, the following steps provide a quick alternative using a Python virtual environment:

1. **Clone the repository**
```bash
git clone https://github.com/your-username/VitaScreen.git
cd VitaScreen
```

2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

The [CDC dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) needs to be available in the [data](data/) folder.

## Usage

### Transform tabular data using IGTD

```bash
export PYTHONPATH=src
python src/data/main.py
```

### Train and evaluate diabetic risk classification model

```bash
export PYTHONPATH=src
python src/train/main.py model=<model_type>
```

Supported model types:

* `catboost`: CatBoost model to be trained on the original data
* `igtd`: CNN model to be trained using IGTD transformed data
* `nctd`: CNN model to be trained using NCTD transformed data

---

## References
* Dataset: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
* [Image Generator for Tabular Data (IGTD)](https://github.com/zhuyitan/IGTD)
