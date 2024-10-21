import coremltools as ct
import numpy as np
import torch
from model import EventDetector


model = EventDetector(pretrain=True, width_mult=1.0, lstm_layers=1, lstm_hidden=256, bidirectional=True, dropout=False)
model.load_state_dict(torch.load('/Users/davidromero/Documents/Capstone/Elaboration F24/ML/golfdb-master/models/swingnet_1800.pth.tar', map_location='cpu')['model_state_dict'])
model.eval()

# Example input for tracing (dummy input with correct shape)
example_input = torch.rand(1, 64, 3, 160, 160)  # Adjust according to your input size

# Step 1: Convert the PyTorch model to TorchScript
scripted_model = torch.jit.trace(model, example_input)

# Step 2: Convert TorchScript model to Core ML using ML Program format for iOS 15+
mlmodel = ct.convert(
    scripted_model, 
    inputs=[ct.TensorType(shape=example_input.shape, dtype=np.float32)],  # Input shape for the model
    source='pytorch',  # Specify that the model source is PyTorch
    minimum_deployment_target=ct.target.iOS15  # iOS 15+
)

# Step 3: Save the Core ML model as .mlpackage
mlmodel.save("GolfSwingDetector.mlpackage")
