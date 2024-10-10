import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch import nn
import torch.optim as optim

# Define the neural network architecture
class SimpleRegressionNet(nn.Module):
    def __init__(self):
        super(SimpleRegressionNet, self).__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Global variables
model = None
train_data = None
test_data = None
X_test = None
y_test = None

# Load dataset
def load_dataset():
    global train_data, test_data, X_test, y_test
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    train_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    for i, (features, label) in enumerate(train_data):
        train_listbox.insert(tk.END, f"Sample {i}: {np.array2string(features, precision=2)}, True-Price: {label:.2f}")

    for i, (features, label) in enumerate(test_data):
        test_listbox.insert(tk.END, f"Sample {i}: {np.array2string(features, precision=2)}, True-Price: {label:.2f}")

# Load model
def load_model():
    global model
    file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch Model", "*.pt")])
    if file_path:
        model = SimpleRegressionNet()
        model.load_state_dict(torch.load(file_path))
        model.eval()
        messagebox.showinfo("Model Loaded", "Model loaded successfully!")

# Train the model
def train_model(epochs):
    global model, train_data
    model = SimpleRegressionNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare training data
    X_train = torch.tensor([features for features, _ in train_data], dtype=torch.float32)
    y_train = torch.tensor([label for _, label in train_data], dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:  # Print every 10 epochs
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Save the trained model
    save_model()

def save_model():
    global model
    file_path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("PyTorch Model", "*.pt")])
    if file_path:
        torch.save(model.state_dict(), file_path)
        messagebox.showinfo("Model Saved", "Model saved successfully!")

# Run inference on selected sample from dataset
def run_inference(selected_sample):
    if model is None:
        messagebox.showwarning("Error", "Please load a model first.")
        return
    selected_features = selected_sample[0]
    true_value = selected_sample[1]
    predicted_value = model(torch.tensor(selected_features, dtype=torch.float32)).item()

    true_label_var.set(f"True Price: ${true_value:.2f}")
    predicted_label_var.set(f"Predicted Price: ${predicted_value:.2f}")

# On list item select for train data
def on_train_select(evt):
    if not train_listbox.curselection():
        return
    index = train_listbox.curselection()[0]
    run_inference(train_data[index])

# On list item select for test data
def on_test_select(evt):
    if not test_listbox.curselection():
        return
    index = test_listbox.curselection()[0]
    run_inference(test_data[index])

# Run inference on custom inputs
def run_manual_inference():
    if model is None:
        messagebox.showwarning("Error", "Please load a model first.")
        return
    
    features = [
        float(medinc_entry.get()), float(houseage_entry.get()), float(averooms_entry.get()), float(avebedrms_entry.get()),
        float(population_entry.get()), float(aveoccup_entry.get()), float(latitude_entry.get()), float(longitude_entry.get())
    ]
    predicted_value = model(torch.tensor(features, dtype=torch.float32)).item()
    predicted_label_var.set(f"Predicted Price: ${predicted_value:.2f}")

# Initialize GUI
root = tk.Tk()
root.title("Inference GUI for Housing Price Regression Model")
root.geometry("1200x600")

# Title
title_label = tk.Label(root, text="Inference GUI for Housing Price Regression Model", font=("Arial", 16), bg="yellow")
title_label.pack(fill=tk.X)

# Frame for buttons and input fields
frame = tk.Frame(root)
frame.pack(pady=10)

# Load Model, Load Dataset and Train Model buttons
load_model_btn = tk.Button(frame, text="Load Model", command=load_model, width=20)
load_model_btn.grid(row=0, column=0, padx=5)

load_dataset_btn = tk.Button(frame, text="Load Dataset", command=load_dataset, width=20)
load_dataset_btn.grid(row=0, column=1, padx=5)

# Train Model button
train_model_btn = tk.Button(frame, text="Train Model (5 epochs)", command=lambda: train_model(5), width=20)
train_model_btn.grid(row=0, column=2, padx=5)

# Input fields for manual inference
medinc_entry = tk.Entry(frame, width=10)
medinc_entry.grid(row=1, column=0)
medinc_label = tk.Label(frame, text="MedInc")
medinc_label.grid(row=2, column=0)

houseage_entry = tk.Entry(frame, width=10)
houseage_entry.grid(row=1, column=1)
houseage_label = tk.Label(frame, text="HouseAge")
houseage_label.grid(row=2, column=1)

averooms_entry = tk.Entry(frame, width=10)
averooms_entry.grid(row=1, column=2)
averooms_label = tk.Label(frame, text="AveRooms")
averooms_label.grid(row=2, column=2)

avebedrms_entry = tk.Entry(frame, width=10)
avebedrms_entry.grid(row=1, column=3)
avebedrms_label = tk.Label(frame, text="AveBedrms")
avebedrms_label.grid(row=2, column=3)

population_entry = tk.Entry(frame, width=10)
population_entry.grid(row=1, column=4)
population_label = tk.Label(frame, text="Population")
population_label.grid(row=2, column=4)

aveoccup_entry = tk.Entry(frame, width=10)
aveoccup_entry.grid(row=1, column=5)
aveoccup_label = tk.Label(frame, text="AveOccup")
aveoccup_label.grid(row=2, column=5)

latitude_entry = tk.Entry(frame, width=10)
latitude_entry.grid(row=1, column=6)
latitude_label = tk.Label(frame, text="Latitude")
latitude_label.grid(row=2, column=6)

longitude_entry = tk.Entry(frame, width=10)
longitude_entry.grid(row=1, column=7)
longitude_label = tk.Label(frame, text="Longitude")
longitude_label.grid(row=2, column=7)

# Run inference on manual input button
run_inference_btn = tk.Button(frame, text="Run Inference Manual Data", command=run_manual_inference, width=20)
run_inference_btn.grid(row=1, column=8)

# Listboxes for train and test data
listboxes_frame = tk.Frame(root)
listboxes_frame.pack(fill=tk.BOTH, expand=True)

train_listbox = tk.Listbox(listboxes_frame, selectmode=tk.SINGLE)
train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
train_listbox.bind('<<ListboxSelect>>', on_train_select)

test_listbox = tk.Listbox(listboxes_frame, selectmode=tk.SINGLE)
test_listbox.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
test_listbox.bind('<<ListboxSelect>>', on_test_select)

# Predicted and True Price Labels
true_label_frame = tk.Frame(root)
true_label_frame.pack(fill=tk.X)

true_label_var = tk.StringVar()
predicted_label_var = tk.StringVar()

true_label = tk.Label(true_label_frame, textvariable=true_label_var, fg="blue", font=("Arial", 14))
true_label.pack(side=tk.LEFT, padx=10, pady=10)

predicted_label = tk.Label(true_label_frame, textvariable=predicted_label_var, fg="red", font=("Arial", 14))
predicted_label.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the GUI
root.mainloop()
