import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms

# Load the trained model
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleNN()
model.load_state_dict(torch.load('mnist_simple_nn.pt'))
model.eval()  # Set model to evaluation mode

# Transform function
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prediction function
def load_image():
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg")])
    if not file_path:
        return

    global loaded_image_path
    loaded_image_path = file_path
    
    image = Image.open(file_path)
    image_tk = ImageTk.PhotoImage(image.resize((150, 150)))
    image_label.config(image=image_tk)
    image_label.image = image_tk
    predict_button.config(state=tk.NORMAL)

# Prediction function
def predict_digit():
    if not loaded_image_path:
        messagebox.showerror("Error", "Please load an image first!")
        return

    image = Image.open(loaded_image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_digit = predicted.item()

    prediction_label.config(text=f"Predicted Digit: {predicted_digit}")

# Create GUI
root = tk.Tk()
root.title("MNIST Digit Predictor")

frame = tk.Frame(root, padx=20, pady=20, bg="#f0f0f0")
frame.pack(fill=tk.BOTH, expand=True)

label = tk.Label(frame, text="MNIST Digit Predictor", font=("Helvetica", 20), bg="#f0f0f0", fg="#333")
label.pack(pady=10)

image_label = tk.Label(frame, bg="#f0f0f0")
image_label.pack(pady=10)

load_button = tk.Button(frame, text="Load Image", command=load_image, bg="#007BFF", fg="white", font=("Helvetica", 14))
load_button.pack(pady=10)

predict_button = tk.Button(frame, text="Predict Digit", command=predict_digit, state=tk.DISABLED, bg="#28a745", fg="white", font=("Helvetica", 14))
predict_button.pack(pady=10)

prediction_label = tk.Label(frame, text="", font=("Helvetica", 16), bg="#f0f0f0", fg="#333")
prediction_label.pack(pady=10)

quit_button = tk.Button(frame, text="Quit", command=root.quit, bg="#6c757d", fg="white", font=("Helvetica", 14))
quit_button.pack(pady=10)

# Global variable to keep track of the loaded image
loaded_image_path = None

root.mainloop()
