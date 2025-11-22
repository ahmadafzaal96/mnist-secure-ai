
# Secure AI Systems – Red & Blue Teaming on MNIST (IDX Format)

This repository contains the complete implementation, experiments, visualizations, and security analysis for a **Convolutional Neural Network (CNN)** trained on the MNIST handwritten digit dataset (IDX format). 
The project demonstrates **real-world adversarial machine learning attacks and defenses**, including:

- ✔️ Data poisoning (backdoor attack)
- ✔️ FGSM adversarial attack (evasion attack)
- ✔️ Adversarial training (defense mechanism)
- ✔️ STRIDE threat modeling
- ✔️ Static Application Security Testing (SAST) using Bandit
- ✔️ Full LaTeX report + Plots + PPT summary

This repository fully satisfies the assignment deliverables for: 
**“Secure AI Systems – Red and Blue Teaming an MNIST Classifier.”**

---

# Repository Structure

```
mnist-secure-ai/
├── src/
│   └── mnist_secure_cnn_idx.py
│
│├── data/
│   └── MNIST
├── report/
│   ├── mnist_secure_ai_report.pdf
│   ├── bandit_report.json
│   └── bandit_report_Fix.json
│
├── requirements.txt
└── README.md
```

⚠️ **Note:** MNIST IDX dataset files are intentionally excluded via `.gitignore`.

---

# 🚀 How to Run This Project

### 1️⃣ Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Place MNIST IDX files in:
```
mnist-secure-ai/data/
```

Required files:
- train-images.idx3-ubyte
- train-labels.idx1-ubyte
- t10k-images.idx3-ubyte
- t10k-labels.idx1-ubyte

### 3️⃣ Run Baseline Training
```bash
python src/mnist_secure_cnn_idx.py --mode baseline --epochs 5
```

### 4️⃣ Run Data Poisoning Attack
```bash
python src/mnist_secure_cnn_idx.py --mode poisoned --epochs 5 --poison_count 100 --poison_target_label 1
```

### 5️⃣ Generate FGSM Adversarial Samples
```bash
python src/mnist_secure_cnn_idx.py --mode adv_only --epochs 5 --epsilon_fgsm 0.3
```

### 6️⃣ Perform Adversarial Training Defense
```bash
python src/mnist_secure_cnn_idx.py --mode adv_training --epochs 5 --epsilon_fgsm 0.3
```

---

# 📊 Performance Summary

## 🔵 Baseline Model
- **Accuracy:** 98.91%
- **Loss:** 0.0400  
- **Inference Time:** ~5.94 m

---

## 🔴 Data Poisoning Attack (Backdoor)
- 100 poisoned samples (7 → 1)
- White square trigger added
- **Clean Test Accuracy:** 98.69% (backdoor remains hidden)
- Strong misclassification when trigger is present


---

## ⚡ FGSM Adversarial Attack (ε = 0.3)
- **Clean Accuracy:** 99.07%
- **FGSM Accuracy:** **17.5%** → severe degradation

---

## 🛡 Adversarial Training Defense

**Before defense**
- FGSM accuracy: **17.5%**

**After defense**
- FGSM accuracy: **81.56%**
- Clean accuracy preserved: **98.96%**


---

# 🛡 STRIDE Threat Model (Summary)

| STRIDE | Threat | Mitigation in This Project |
|--------|--------|----------------------------|
| **S – Spoofing** | Fake data sources | Controlled IDX loading, no remote inputs |
| **T – Tampering** | Poisoned data | Explicit poisoning module demonstrates risk; integrity checks prevent accidental tampering |
| **R – Repudiation** | Hidden malicious actions | All runs require explicit modes (baseline/poisoned/adv_only/adv_training) |
| **I – Information Disclosure** | Leaking model/data | All operations are offline/local |
| **D – DoS** | Adversarial input overload | Inference-time benchmarking under attack performed |
| **E – Elevation of Privilege** | Unauthorized retraining | Clear separation of training modes and poisoning logic |

Full STRIDE analysis is available in the LaTeX report.

---

# 🔍 Static Application Security Testing (SAST)

Tool Used: **Bandit**

### Results:
- Low-severity: Use of `assert` (fixed to `ValueError`)
- No medium/high issues in our implementation
- Reports included:
  - bandit_report.json
  - bandit_report_Fix.json

---

# 🏁 Conclusion

This project demonstrates that:

- High accuracy **does not guarantee** robustness.
- A small poisoning set (100 samples) can implant a stealthy backdoor.
- FGSM adversarial noise can **break** the classifier completely.
- Adversarial training improves robustness from **17.5% → 81.56%**.
- STRIDE analysis and SAST enhance the overall security posture.

---

# ✍️ Author

- Name : Afzaal Ahmad
- Department of Computer Science  
- Indian Institute of Technology Hyderabad
