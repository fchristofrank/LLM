🧠 Simple Bigram Language Model in PyTorch
A minimal implementation of a Bigram Language Model using PyTorch.

This project demonstrates the core principles of language modeling by predicting the next token based solely on the current token. It serves as an ideal starting point for understanding autoregressive language models before diving into more advanced architectures like Transformers.

Inspired by Andrej Karpathy’s “Let’s build GPT” tutorial.

📚 Overview
This Bigram Language Model:

Uses a single nn.Embedding layer as a lookup table for next-token prediction.

Trains on character-level inputs using cross-entropy loss.

Generates new text autoregressively by sampling one token at a time.

Is simple, yet effective, for educational purposes.

🚀 Features
🔢 Token-level embedding for input-output mapping

🔁 Autoregressive sampling for text generation

🧪 Minimal training loop with torch.nn and torch.optim

🧼 Clean, readable, and beginner-friendly code

📦 No dependencies beyond PyTorch

🛠️ Installation
bash
Copy
Edit
git clone https://github.com/your-username/simple-bigram-model
cd simple-bigram-model
pip install torch
📄 Usage
1. Prepare Input
Put a plain text file named input.txt in the repo directory.

Example:

bash
Copy
Edit
echo "hello world" > input.txt
2. Train the Model
bash
Copy
Edit
python bigram.py
3. Generate Text
The model will print generated samples during and after training.

📁 File Structure
graphql
Copy
Edit
📦simple-bigram-model
 ┣ 📄input.txt           # Training data (plain text)
 ┣ 📄bigram.py           # Main training & generation script
 ┗ 📄README.md           # Project documentation
🔍 Example Output
yaml
Copy
Edit
Generated sample:
tht tnd t tnd tot tnnt nttddhdnt hthdttdt...
(Not coherent yet—but that’s the beauty of starting simple!)

🧠 Learn More
This repo is educational in nature and best understood alongside the original tutorial:

📺 Karpathy’s “Let’s build GPT”

🤝 Contributing
Contributions, ideas, and improvements are welcome! Feel free to open an issue or submit a PR.

📜 License
MIT License

