# Adversial-training

**Adversarial Training Framework for Large Language Models (LLMs)**

This project implements an adversarial training framework designed to enhance the robustness of large language models (LLMs) against adversarial attacks. It leverages the bert-base-uncased pretrained model, fine-tuned on the SST-2 sentiment classification dataset. TextAttack was used to generate adversarial examples, and a two-phase training approach was implemented to systematically improve the model’s robustness.

📈 **Project Summary**  
	•	**Objective:** Improve LLM robustness using adversarial training strategies.  
	•	**Model Used:** bert-base-uncased (Hugging Face)  
	•	**Dataset:** SST-2 (Stanford Sentiment Treebank)  
	•	**Tools & Libraries:** TensorFlow, Hugging Face Transformers, TextAttack, Google Colab Pro  
	•	**Key Results:**  
			98% Adversarial Accuracy  
			89% Overall Accuracy  

 📦 **Features**  
	•	✅ **BERT Fine-Tuning:** Fine-tuned the bert-base-uncased model on SST-2.  
	•	✅ **Adversarial Generation:** Generated advanced adversarial examples using the TextAttack library.  
	•	✅ **Two-Phase Training Approach:**  
	    		Phase 1: Trained on clean data, tested on adversarial examples.  
	        	Phase 2: Trained on a combined dataset of clean and adversarial examples.  
	•	✅ **Performance Analysis:** Evaluated robustness improvement across both phases.  
	•	✅ **Cloud Integration:** Used Google Colab Pro for scalable model training and testing.  
 
 📊 **Project Structure**
```
├── data/                      # Contains SST-2 dataset
├── models/                    # Pre-trained and fine-tuned model checkpoints
├── src/                       # Source code for model training and evaluation
│   ├── adversarial_training.py
│   ├── data_preprocessing.py
│   └── loss_functions.py
├── results/                   # Performance metrics and logs
├── notebooks/                 # Colab notebooks for experimentation
├── requirements.txt           # Dependencies for the project
└── README.md                  # This file
```
📚 **Methodology**

Step 1: Dataset Preparation
	•	Downloaded and preprocessed the SST-2 dataset for binary sentiment classification.

Step 2: Model Fine-Tuning
	•	Fine-tuned the bert-base-uncased model using TensorFlow and Hugging Face Transformers.

Step 3: Adversarial Example Generation
	•	TextAttack was used to generate adversarial examples by perturbing input text while preserving its semantic meaning.

Step 4: Two-Phase Training
		Phase 1: Trained the model using clean data and evaluated against adversarial examples.
		Phase 2: Trained on a combined dataset (clean + adversarial examples) and evaluated final performance.

📈 **Results**
| Metric                  | Phase 1 (Clean Data) | Phase 2 (Combined Data) |
|-------------------------|----------------------|------------------------|
| Adversarial Accuracy    | 85%                 | 98%                   |
| Overall Accuracy        | 87%                 | 89%                   |
| Loss Reduction Rate     | High                | Significant Drop      |

🚀 **Getting Started**

**Prerequisites:**  

Ensure you have the following installed:  
	•	Python 3.x  
	•	TensorFlow  
	•	Hugging Face Transformers  
	•	TextAttack Library  
	•	Google Colab Pro (optional for cloud-based execution)  

**Installation:**
```
git clone https://github.com/yourusername/adversarial-training-llms.git
cd adversarial-training-llms
pip install -r requirements.txt
```
Run the Training:
```
python src/adversarial_training.py --dataset data/sst2 --epochs 10 --model bert-base-uncased
```
📊 **Performance Visualization**  
	•	Performance metrics, loss curves, and adversarial accuracy plots are available in the results/ directory.

 🤝 **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request for review.
