# Breast Cancer Classification using Neural Network

A deep learning project that classifies breast cancer tumors as **Malignant** or **Benign** using a Neural Network built with TensorFlow and Keras, trained on the Wisconsin Breast Cancer Dataset.

---

## 📌 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early and accurate detection is critical for effective treatment. This project builds a binary classification model that predicts whether a tumor is malignant (cancerous) or benign (non-cancerous) based on 30 clinical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

---

## 🗂️ Dataset

- **Source:** [sklearn.datasets.load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- **Samples:** 569
- **Features:** 30 numerical features (e.g., radius, texture, perimeter, area, smoothness, etc.)
- **Target:**
  - `1` → Benign
  - `0` → Malignant

---

## 🧠 Model Architecture

The neural network is built using Keras Sequential API:

| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| Input | Flatten | 30 | — |
| Hidden | Dense | 20 | ReLU |
| Output | Dense | 2 | Sigmoid |

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 10
- **Validation Split:** 10%

---

## 🔄 Workflow

1. **Data Collection** — Load the Breast Cancer dataset from scikit-learn
2. **Exploratory Data Analysis** — Shape, info, null checks, statistical summary, label distribution
3. **Preprocessing** — Separate features (X) and labels (Y); split into 80% train / 20% test
4. **Standardization** — Apply `StandardScaler` to normalize feature values
5. **Model Building** — Define and compile the Neural Network
6. **Training** — Fit the model and track accuracy/loss over epochs
7. **Evaluation** — Evaluate on test data; visualize training vs. validation curves
8. **Prediction** — Build a predictive system for new input data

---

## 📊 Visualizations

The notebook generates:
- **Accuracy curve** — Training vs. Validation accuracy over epochs
- **Loss curve** — Training vs. Validation loss over epochs

---

## 🛠️ Technologies Used

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `matplotlib` | Visualization |
| `scikit-learn` | Dataset, train/test split, StandardScaler |
| `tensorflow` / `keras` | Neural Network building and training |

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kaleem205/Breast-Cancer-Classification-with-a-simple-Neural-Network-NN-
   cd breast-cancer-classification
   ```

2. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook impelementation.ipynb
   ```

---

## 🔍 Making a Prediction

The notebook includes a ready-to-use predictive system. Pass a tuple of 30 feature values and the model will output whether the tumor is **Malignant** or **Benign**:

```python
input_data = (11.76, 21.6, 74.72, 427.9, ...)  # 30 feature values

# Preprocess and predict
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
prediction_label = [np.argmax(prediction)]

if prediction_label[0] == 0:
    print('The tumor is Malignant')
else:
    print('The tumor is Benign')
```

---

## 📁 Project Structure

```
breast-cancer-classification/
│
├── impelementation.ipynb   # Main Jupyter Notebook
└── README.md               # Project documentation
```

---

## 🙏 Acknowledgements

- Dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) via scikit-learn
- Built with [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)