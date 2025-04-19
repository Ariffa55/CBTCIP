import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Toy Dataset
pairs = [
    ["hello", "hi"],
    ["hi", "hello"],
    ["how are you", "i am fine, thank you"],
    ["what is your name", "i am a chatbot"],
    ["who created you", "i was created by a developer"],
    ["what can you do", "i can chat with you"],
    ["tell me a joke", "why don’t scientists trust atoms? because they make up everything!"],
    ["what is the weather like", "i don’t have access to real-time weather yet"],
    ["bye", "goodbye"],
    ["see you later", "take care"]
]

# Vocabulary
words = set(word for pair in pairs for sentence in pair for word in sentence.split())
word2idx = {word: i+2 for i, word in enumerate(words)}
word2idx["<PAD>"] = 0
word2idx["<EOS>"] = 1
idx2word = {i: w for w, i in word2idx.items()}

# Utils
def sentence_to_tensor(sentence):
    idxs = [word2idx[word] for word in sentence.split()]
    idxs.append(word2idx["<EOS>"])
    return torch.tensor(idxs, dtype=torch.long).unsqueeze(1)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded)
        return hidden

# Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.rnn(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Training Loop
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder(input_tensor)

    decoder_input = torch.tensor([[word2idx["<EOS>"]]])
    decoder_hidden = encoder_hidden

    loss = 0
    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_tensor.size(0)

# Model Setup
hidden_size = 256
encoder = Encoder(len(word2idx), hidden_size)
decoder = Decoder(hidden_size, len(word2idx))

encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training
for epoch in range(1000):
    pair = random.choice(pairs)
    input_tensor = sentence_to_tensor(pair[0])
    target_tensor = sentence_to_tensor(pair[1])
    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Chat
def evaluate(sentence):
    with torch.no_grad():
        input_tensor = sentence_to_tensor(sentence)
        hidden = encoder(input_tensor)
        decoder_input = torch.tensor([[word2idx["<EOS>"]]])

        decoded_words = []
        for _ in range(10):
            output, hidden = decoder(decoder_input, hidden)
            topv, topi = output.topk(1)
            if topi.item() == word2idx["<EOS>"]:
                break
            else:
                decoded_words.append(idx2word[topi.item()])
                decoder_input = topi.squeeze().detach().unsqueeze(0)
        return ' '.join(decoded_words)


# Interactive Chat Loop
print("\n=== Chat with your bot (type 'exit' to stop) ===")
while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        print("Bot: Goodbye! 👋")
        break
    
    response = evaluate(user_input)
    print("Bot:", response)
    
