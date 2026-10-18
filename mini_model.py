
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# 1. Tiny training corpus
text = (
    "hello world. hello there. "
    "hello friend. how are you? "
    "how are we? hello world. "
) * 100

# 2. Character vocabulary and token IDs
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encoded = torch.tensor([stoi[ch] for ch in text])
vocab_size = len(chars)

# 3. Create input-output examples
# Three characters predict the next character.
context_size = 3
X, y = [], []

for i in range(context_size, len(encoded)):
    X.append(encoded[i - context_size:i])
    y.append(encoded[i])

X = torch.stack(X)
y = torch.stack(y)

print("Input dataset shape:", X.shape)
print("Target dataset shape:", y.shape)
print("Vocabulary size:", vocab_size)

# 4. Define a small neural network
class NextCharModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 16)
        self.hidden = nn.Linear(context_size * 16, 64)
        self.output = nn.Linear(64, vocab_size)

    def forward(self, x):
        # x: (batch_size, 3)
        x = self.embedding(x)
        print("Embedding shape:", x.shape)

        # (batch_size, 3, 16) -> (batch_size, 48)
        x = x.reshape(x.size(0), -1)
        print("Flattened shape:", x.shape)

        x = torch.tanh(self.hidden(x))
        print("Hidden shape:", x.shape)

        x = self.output(x)
        print("Output shape:", x.shape)

        return x  # Raw logits

model = NextCharModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# 5. Training
for epoch in range(10):
    logits = model(X)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}: loss = {loss.item():.4f}")

# 6. Inference: generate characters without updating weights
@torch.no_grad()
def generate(start="hel", max_new_chars=60):
    model.eval()

    unknown_chars = [ch for ch in start if ch not in stoi]
    if unknown_chars:
        raise ValueError(
            f"Characters not in vocabulary: {unknown_chars}"
        )

    if len(start) < context_size:
        start = " " * (context_size - len(start)) + start

    result = list(start)
    context = [stoi[ch] for ch in result[-context_size:]]

    for _ in range(max_new_chars):
        x = torch.tensor([context])
        logits = model(x)[0]
        probabilities = torch.softmax(logits, dim=-1)

        # Greedy decoding: choose the highest-probability character.
        next_id = torch.argmax(probabilities).item()
        next_char = itos[next_id]

        result.append(next_char)
        context = context[1:] + [next_id]

    return "".join(result)

# Disable shape debugging during generation.
# Set this to False after inspecting the training output.
print("\nGenerated text:")
print(generate())
