# CBTCIP

### 🤖 Simple Seq2Seq Chatbot (PyTorch)

This project is a minimalistic chatbot implementation built using PyTorch, featuring a sequence-to-sequence (Seq2Seq) model with GRU-based encoder and decoder networks. It is trained on a small toy dataset of predefined conversational pairs and demonstrates the basic working of neural conversational agents.

** Model Architecture:**
- **Encoder:** GRU-based RNN that processes input sentences word-by-word.
- **Decoder:** GRU-based RNN that generates output sentences autoregressively.
- **Embedding Layer:** Shared between both encoder and decoder for converting word indices into dense vector representations.
- **Teacher Forcing:** Used during training to accelerate convergence.

** Dataset:**
- A toy dataset of hardcoded input-output pairs.
- Vocabulary is automatically constructed from the dataset with special tokens (`<PAD>`, `<EOS>`).

** Features:**
- End-to-end training with manual word-token mapping
- Custom data preprocessing and input tensorization
- Evaluation with greedy decoding
- Interactive terminal-based chat interface
- Minimal dependencies and beginner-friendly structure

** Tech Stack:**
- Python 3.x  
- PyTorch  
- NumPy

** Getting Started:**
1. Run the script to train the model (training runs for 1000 epochs on a toy dataset).
2. Chat with the bot in the terminal via an interactive loop.
3. Type `exit` to quit the chat.

** Future Improvements:**
- Expand the dataset for better generalization.
- Replace toy dataset with real-world dialogue data.
- Add attention mechanism for improved context tracking.
- Save and load trained models for reuse.

