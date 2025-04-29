
# AI-Driven Loan Approval using PSO-Enhanced ANN & GBM

This repository contains the implementation of an AI-based system for predicting loan approval using optimized machine learning models — Artificial Neural Networks (ANN) and Gradient Boosting Machines (GBM) — enhanced through Particle Swarm Optimization (PSO).

## 📌 Project Overview

In the financial sector, loan approval prediction plays a critical role in risk management and customer service. This project explores the application of machine learning to automate and enhance loan approval prediction accuracy.

### 🔍 Key Highlights:
- **Data Preprocessing**: Feature engineering, handling missing values, encoding, and scaling.
- **Model Building**:
  - ANN with 4-layer architecture and dropout regularization.
  - GBM with decision tree base learners.
- **Optimization**: Hyperparameter tuning using Particle Swarm Optimization (PSO).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Cross-validation.

---

## 📁 Project Structure

```
├── PSO.ipynb                        # Jupyter Notebook containing full implementation
├── AI-Driven Loan Approval.pdf     # Full research paper with methodology & results
├── README.md                       # Project description and documentation
```

---

## 🛠️ Technologies Used

- Python 3
- Scikit-learn
- XGBoost
- Keras (TensorFlow backend)
- SMOTE (Imbalanced-learn)
- NumPy, Pandas, Matplotlib, Seaborn

---

## 🔧 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/loan-approval-psonn-gbm.git
   cd loan-approval-psonn-gbm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open and Run the Notebook**
   ```bash
   jupyter notebook PSO.ipynb
   ```

---

## 📊 Results

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| ANN (No PSO)     | 84%      | 75.5%     | 78.5%  | 77.0%    |
| ANN (With PSO)   | 89%      | 77.5%     | 80.5%  | 78.5%    |
| GBM (No PSO)     | 89%      | 89.0%     | 73.5%  | 78.0%    |
| GBM (With PSO)   | 89%      | 89.5%     | 74.0%  | 78.5%    |

---

## 💡 Future Work

- Incorporating behavioral and social data into the model.
- Improving interpretability of black-box models.
- Exploring hybrid model architectures combining ANN, GBM, and SVM.
- Testing alternate optimization algorithms like Genetic Algorithm or Bayesian Optimization.

---

## 👨‍💻 Authors

- Akash Saraswat  
- Anushika Gupta  
- Divya Pandey *(Corresponding Author)* – [divyapandey1113@gmail.com](mailto:divyapandey1113@gmail.com)  
- Ishitta  

All from: B.Tech III Year, Centre for Artificial Intelligence, MITS Gwalior, India.



Would you like help creating a `requirements.txt` or the license file as well?
