# Adversial-training

   
**Adversarial Training Framework for Large Language Models (LLMs)**

This project implements an adversarial training framework designed to enhance the robustness of large language models (LLMs) against adversarial attacks. It leverages the bert-base-uncased pretrained model, fine-tuned on the SST-2 sentiment classification dataset. TextAttack was used to generate adversarial examples, and a two-phase training approach was implemented to systematically improve the model’s robustness.

## 📈 Project Summary
- **Objective:** Improve LLM robustness using adversarial training strategies.  
- **Model Used:** `bert-base-uncased` (Hugging Face)  
- **Dataset:** SST-2 (Stanford Sentiment Treebank)  
- **Tools & Libraries:** TensorFlow, Hugging Face Transformers, TextAttack, Google Colab Pro  
- **Key Results:**  
   - ✅ **98% Adversarial Accuracy**  
   - ✅ **89% Overall Accuracy**


## 📦 Features
- **BERT Fine-Tuning:** Fine-tuned the `bert-base-uncased` model on the SST-2 dataset for binary sentiment classification.  
- **Adversarial Generation:** Generated advanced adversarial examples using the `TextAttack` library.  
- **Two-Phase Training Approach:**  
   - **Phase 1:** Trained on clean data, tested on adversarial examples.  
   - **Phase 2:** Trained on a combined dataset of clean and adversarial examples.  
- **Performance Analysis:** Evaluated robustness improvement across both phases with significant adversarial accuracy gains.  
- **Cloud Integration:** Leveraged **Google Colab Pro** for scalable model training and testing.  
 
## 📊 Project Workflow
1. **Data Preparation:**
   - Loaded SST-2 dataset using Hugging Face `datasets` library.
   - Preprocessed data for binary sentiment classification tasks.

2. **Baseline Fine-Tuning:**
   - Fine-tuned the `bert-base-uncased` model on the SST-2 dataset using Hugging Face's `Trainer` API.
   - Achieved initial clean test accuracy: **85.3%**

3. **Adversarial Example Generation:**
   - Generated adversarial examples using `TextAttack` with the `TextFoolerJin2019` attack method.
   - Expanded with additional lightweight attacks:
     - Character-level perturbations (typos, swaps)
     - Back-translation
     - Word order shuffling
     - Contrastive examples (minimal pairs with the same label)

4. **Adversarial Fine-Tuning:**
   - Combined the clean dataset with generated adversarial examples.
   - Fine-tuned the model on the combined dataset with a custom adversarial loss function.
   - Improved **Adversarial Accuracy**: 🔺 **58% → 98%**

📈 **Results**
| Metric                  | Phase 1 (Clean Data)| Phase 2 (Combined Data) |
|-------------------------|---------------------|------------------------|
| Adversarial Accuracy    | 58%                 | 98%                   |
| Overall Accuracy        | 87%                 | 89%                   |
| Loss Reduction Rate     | High                | Significant Drop      |

---

🚀 **Getting Started**

**Prerequisites:**  

Ensure you have the following installed:  
	•	Python 3.x  
	•	TensorFlow  
	•	Hugging Face Transformers  
	•	TextAttack Library  
	•	Google Colab Pro (optional for cloud-based execution)  


## 📦 Files in the Repository
- **`Adversarial_Training_SST2_BERT_Colab.ipynb`**: Colab notebook containing the full training pipeline, including data loading, fine-tuning, adversarial generation, and evaluation.
- **`requirements.txt`**: Python dependencies for running the notebook locally.
- **`README.md`**: This documentation file.

---

## 🚀 How to Run the Notebook
1. **Open in Colab:**
   - [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](your-colab-notebook-link-here)
2. **Run on Local Machine:**
   ```bash
   git clone <repository-url>
   cd <repo-directory>
   pip install -r requirements.txt
   jupyter notebook```

## 📈 Future Work
- **Expanded Attack Strategies:** Implement additional attack methods like **HotFlip**, **PWWS**, and **DeepWordBug** for a broader adversarial evaluation.  
- **Advanced Training Techniques:** Apply **Continuous Adversarial Training (CAT)** and **Continuous Adversarial Preference Optimization (CAPO)** for improved robustness and training efficiency.  
- **Contrastive Learning:** Use a dataset combining **positive (safe)** and **negative (harmful)** examples with contrastive loss to help the model effectively distinguish harmful from non-harmful inputs.

---

## ⭐ Acknowledgments
- 🙌 **Hugging Face** for the `Transformers` and `Datasets` libraries.  
- 🎯 **TextAttack** for the adversarial attack tools.  
- 📊 **SST-2 Dataset Authors** for their contributions to sentiment analysis research.  
