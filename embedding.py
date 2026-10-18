
# Training a simple model manually

# Training example
x = 2.0
y = 6.0

# Initial trainable parameters
w = 2.0
b = 0.0

# Training configuration
learning_rate = 0.1
epochs = 10

for epoch in range(epochs):

    # 1. FORWARD PASS: make a prediction
    y_pred = w * x + b

    # 2. LOSS: measure prediction error
    loss = (y_pred - y) ** 2

    # 3. BACKPROPAGATION: calculate gradients
    # dL/dw = 2 * (prediction - target) * input
    dw = 2 * (y_pred - y) * x

    # dL/db = 2 * (prediction - target)
    db = 2 * (y_pred - y)

    # 4. GRADIENT DESCENT: update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    print(
        f"Epoch {epoch + 1}: "
        f"Prediction={y_pred:.4f}, "
        f"Loss={loss:.4f}, "
        f"Gradient_w={dw:.4f}, "
        f"Weight={w:.4f}"
    )

print("\nFinal prediction:", w * x + b)
print("Final weight:", w)
print("Final bias:", b)
