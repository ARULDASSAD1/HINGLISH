# Hinglish Voice-AI Fine-Tuning (GPT-2)

## 🎯 Objective

To fine-tune a GPT-2 model for conversational Hinglish (code-switched) dialogue using Hugging Face due to budget constraints (OpenAI API not used).

## 📁 Dataset

Collected 80+ conversational Hinglish examples covering casual, college, food, weekend plans, etc.

Example:
{"prompt":"User: Tu college aa raha hai aaj?\nAssistant:","completion":"Haan yaar, aaj attendance full karni hai."}

## 🧠 Model

Used `gpt2` base model with 10 epochs.

- **Why GPT-2?**: Open-source, no API cost.
- **Tokenizer**: GPT2Tokenizer
- **Epochs**: 10
- **Batch Size**: 4
- **Top_p**: 0.65 to reduce repetition
- **Temperature**: 0.7 for variation

## 🧪 Sample Outputs

- **Prompt**: `User: Tum kis subject mein achhe ho?\nAssistant:`
- **Output**: `Main computer science mein best hoon, tumhara kya scene hai?`

## 📝 Setup

```bash
pip install transformers datasets
Then run:

python fine_tune.py to train

python inference.py to generate replies

✅ Note
As per approval from ConversalLabs, we used Hugging Face fine-tuning instead of OpenAI API due to budget constraints.

---
