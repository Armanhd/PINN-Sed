
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import tomllib as toml
#%%
# Load config

toml_path=r'C:\Users\haddadchia\Dropbox\Neshat\Machine Learning\PINN_Sed\config.toml'
with open(toml_path, "rb") as f:
    config = toml.load(f)

# -------------------------
# Paths
# -------------------------
PINN_dir = config["paths"]["PINN_dir"]
loss_directory = config["paths"]["loss_directory"]
output_csv_name = config["paths"]["output_csv_name"]
input_file = config["paths"]["input_file"]

# -------------------------
# Data settings
# -------------------------
sheet = config["data"]["sheet"]
columns = config["data"]["columns"]

# -------------------------
# Read Excel
# -------------------------
c_df = pd.read_excel(
    os.path.join(PINN_dir, input_file),
    sheet_name=sheet,
    header=0,
    usecols=columns
)

# -------------------------
# Scaling
# -------------------------
div = config["scaling"]["c_divisor"]

c1_input = c_df['ctop_MAI'] / div
c2_input = c_df['ctop_NGA'] / div
c3_input = c_df['ctop_TCOL'] / div

t_steps = len(c1_input)

# -------------------------
# X domains
# -------------------------
def compute_domain(case):
    start = case["start"]
    dx = case["dx"]
    steps = case["steps"]
    end = (steps - 1) * dx
    return start, end, steps

x1_start, x1_end, x_steps = compute_domain(config["x_domain"]["case1"])
x2_start, x2_end, x_steps = compute_domain(config["x_domain"]["case2"])
x3_start, x3_end, x_steps = compute_domain(config["x_domain"]["case3"])

# -------------------------
# Debug print
# -------------------------
print("t_steps:", t_steps)
print("x1:", x1_start, x1_end, x_steps)
print("x2:", x2_start, x2_end, x_steps)
print("x3:", x3_start, x3_end, x_steps)
# dt=300 s, t_steps, (e.g. 40 steps from 0 --> 11700 s
t_start, t_end= 0, (t_steps-1) * 300  # t domain

# Create the x and t arrays

x1= np.linspace(x1_start, x1_end, x_steps)[:, None]
x2= np.linspace(x2_start, x2_end, x_steps)[:, None]
x3= np.linspace(x3_start, x3_end, x_steps)[:, None]


t = np.linspace(t_start, t_end, t_steps)[:, None]

#%% Define concentration

# c = np.zeros((12, 40))

c1=np.zeros((x_steps, t_steps))
c1[:,0]=c1_input[0]
c1[0,:]=c1_input

c2=np.zeros((x_steps, t_steps))
c2[:,0]=c2_input[0]
c2[0,:]=c2_input

c3=np.zeros((x_steps, t_steps))
c3[:,0]=c3_input[0]
c3[0,:]=c3_input
#delete unused terms for the rest of the code
del c_df, c1_input, c2_input, c3_input 

#%%#%% Insert h, q, Q, slope, streampower critical as input to the model -  xls (saved) 
from functions_sed_pinn import read_h_q_xls

h1,h2,h3,q1,q2,q3,Q1,Q2,Q3,Strmpowcri1,Strmpowcri2,Strmpowcri3,slope1,slope2,slope3=read_h_q_xls (PINN_dir,input_file )

h1_start=np.min (h1)
h1_end=np.max (h1)
h2_start=np.min (h2)
h2_end=np.max (h2)
h3_start=np.min (h3)
h3_end=np.max (h3)
q1_start=np.min (q1)
q1_end=np.max (q1)
q2_start=np.min (q2)
q2_end=np.max (q2)
q3_start=np.min (q3)
q3_end=np.max (q3)
Q1_start=np.min (Q1)
Q1_end=np.max (Q1)
Q2_start=np.min (Q2)
Q2_end=np.max (Q2)
Q3_start=np.min (Q3)
Q3_end=np.max (Q3)
strmpowcri1_start=np.min (Strmpowcri1)
strmpowcri1_end=np.max (Strmpowcri1)
strmpowcri2_start=np.min (Strmpowcri2)
strmpowcri2_end=np.max (Strmpowcri2)
strmpowcri3_start=np.min (Strmpowcri3)
strmpowcri3_end=np.max (Strmpowcri3)

#%% calculate re-entrainment rate [kg/m2/s] and Stream power

Cri1 = config["constants"]["Cri1"]
Cri2 = config["constants"]["Cri2"]
Cri3 = config["constants"]["Cri3"]
ki = config["constants"]["ki"]
Fi = config["constants"]["Fi"]
rho = config["constants"]["rho"]
g = config["constants"]["g"]
rhos = config["constants"]["rhos"]


Strmpow1=rho*g*Q1*slope1 # stream power
ryd1=((Fi*(Strmpow1-(Cri1*Strmpowcri1))
                /(ki*g*h1)))*(rhos/(rhos-rho))
Strmpow2=rho*g*Q2*slope2 # stream power
ryd2=((Fi*(Strmpow2-(Cri2*Strmpowcri2))
                /(ki*g*h2)))*(rhos/(rhos-rho))
Strmpow3=rho*g*Q3*slope3 # stream power
ryd3=((Fi*(Strmpow3-(Cri3*Strmpowcri3))
                /(ki*g*h3)))*(rhos/(rhos-rho))

#%% Create Neural Network
# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # "cuda"  means computations will run on the GPU
# device = torch.device('cuda')

# Define a simple feedforward neural network
class NN(nn.Module):   # "torch.nn.Module" is the base class for all neural network models in PyTorch
    # Initialize the model and its parameters
    def __init__(self,
                 input_size,
                 output_size,
                 num_layers,     # Number of hidden layers.
                 hidden_size,    # Number of neurons in each hidden layer
                 lb,             # lower bound for input scaling (used for normalization)
                 ub):            # upper bound for input scaling (used for normalization)
        super(NN, self).__init__()  # Calls the constructor of the parent nn.Module to properly initialize the object
        
        layers = []             # A list that stores the neural network layers
        layers.append(nn.Linear(input_size, hidden_size))       # Creates a fully connected (dense) layer from input_size to hidden_size
        layers.append(nn.Tanh())        # Applies the hyperbolic tangent activation function to introduce non-linearity
        
        for _ in range(num_layers - 1):     
            # Adds num_layers - 1 hidden layers, each with hidden_size neurons and Tanh activation.
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_size, output_size))      # Creates a fully connected (dense) layer from hidden_size to output_size
        # # Output layer with Softplus activation (Arman)
        # layers.append(nn.Softplus())  # Ensures output is always c > 0
        
        # Combines all layers into a sequential container, allowing easy 
        # execution of forward passes through all layers.
        self.network = nn.Sequential(*layers)  # The output of one layer becomes the input of the next.
        
        # Store the lower and upper bounds for input normalization. 
        # These will be used to scale the input values to a range of [-1, 1]
        self.lb = lb
        self.ub = ub 
    
    def forward(self, x):       # Defines the forward pass of the network, which specifies how the input propagates through the layers
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0 # Normalization (input scaling)
        return self.network(x)      # Passes the scaled input x through the defined layers in self.network
    
#%% PDE residual calculation
c_alpha = config["constants"]["c_alpha"]
di = config["constants"]["di"]         # particle size in [m] for size class
def pde_fn(c, x, t, h, q, ryd):
    # h = 0.2
    # q = 0.03
    k = 250
    # ryd = 0.1 * c
    # ded = 0.08 * c
    from functions_sed_pinn import ded
    
    ded=ded(c,di, c_alpha)
    
    # To compute the gradients of c with respect to its inputs (x or t), 
    # PyTorch needs to know how to handle the vector-Jacobian product. 
    # Using a vector of ones ensures all derivatives are equally weighted and 
    # effectively sums the contributions if c is a tensor.
    c_t, c_x = torch.autograd.grad(c, [t, x], grad_outputs=torch.ones_like(c), create_graph=True)
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    # Add h from t and x into the model 
    # h=h
    # q=q
    residual = h * c_t + q * c_x - k * c_xx - ryd + ded
    return residual

#%%
# Initialize the model
input_size = 3 # which are x and t
output_size = 1 # which is c

x1_min, x1_max = x1.min(), x1.max()
x2_min, x2_max = x2.min(), x2.max()
x3_min, x3_max = x3.min(), x3.max()
t_min, t_max = t.min(), t.max()

# upper and lower boundary bsed on x and t
lb1 = torch.tensor([x1_min, t_min, h1_start], dtype=torch.float32).to(device)
ub1 = torch.tensor([x1_max, t_max, h1_end], dtype=torch.float32).to(device)
lb2 = torch.tensor([x2_min, t_min, h2_start], dtype=torch.float32).to(device)
ub2 = torch.tensor([x2_max, t_max, h2_end], dtype=torch.float32).to(device)
lb3 = torch.tensor([x3_min, t_min, h3_start], dtype=torch.float32).to(device)
ub3 = torch.tensor([x3_max, t_max, h3_end], dtype=torch.float32).to(device)

# Model definition
model1 = NN(input_size, output_size, num_layers=5, hidden_size=128, lb=lb1, ub=ub1).to(device)
model2 = NN(input_size, output_size, num_layers=5, hidden_size=128, lb=lb2, ub=ub2).to(device)
model3 = NN(input_size, output_size, num_layers=5, hidden_size=128, lb=lb3, ub=ub3).to(device)


# Define loss functions
# In PINNs, the loss function typically combines:
# The data loss: Compares the network's prediction (c) with observed data.
# The PDE residual loss: Ensures that the network satisfies the governing equations.
criterion = nn.MSELoss() #loss function in PyTorch that computes the Mean Squared Error (MSE) between the predicted and target values.

# Optimizer
# Initializes the optimizer, which updates the neural network's parameters during training
optimizer1 = optim.Adam(model1.parameters(), lr=0.001) #lr= learning rate
optimizer2 = optim.Adam(model2.parameters(), lr=0.001) #lr= learning rate
optimizer3 = optim.Adam(model3.parameters(), lr=0.001) #lr= learning rate

# Training data preparation
# Converting the input data (x and t) into PyTorch tensors suitable for training
x1_tensor = torch.tensor(x1.flatten(), dtype=torch.float32, requires_grad=True).to(device)
x2_tensor = torch.tensor(x2.flatten(), dtype=torch.float32, requires_grad=True).to(device)
x3_tensor = torch.tensor(x3.flatten(), dtype=torch.float32, requires_grad=True).to(device)

t_tensor = torch.tensor(t.flatten(), dtype=torch.float32, requires_grad=True).to(device)

h1_tensor=torch.tensor(h1, dtype=torch.float32).to(device)
q1_tensor=torch.tensor(q1, dtype=torch.float32).to(device)
h2_tensor=torch.tensor(h2, dtype=torch.float32).to(device)
q2_tensor=torch.tensor(q2, dtype=torch.float32).to(device)
h3_tensor=torch.tensor(h3, dtype=torch.float32,).to(device)
q3_tensor=torch.tensor(q3, dtype=torch.float32,).to(device)


ryd1_tensor=torch.tensor(ryd1, dtype=torch.float32).to(device)
ryd2_tensor=torch.tensor(ryd2, dtype=torch.float32).to(device)
ryd3_tensor=torch.tensor(ryd3, dtype=torch.float32).to(device)

# Create the meshgrid
# using input tensors of x and t
X1, T = torch.meshgrid(x1_tensor, t_tensor, indexing='ij')
X2, T = torch.meshgrid(x2_tensor, t_tensor, indexing='ij')
X3, T = torch.meshgrid(x3_tensor, t_tensor, indexing='ij')


# Preparing a 2D tensor of inputs (pairs of x and t) 
# torch.cat([...]): Concatenates X and T into a 2D tensor, where each row is a pair [x, t]
inputs1 = torch.cat([X1.flatten()[:, None], T.flatten()[:, None], q1_tensor.flatten()[:, None]], dim=1) # concatenate along columns
inputs2 = torch.cat([X2.flatten()[:, None], T.flatten()[:, None], q2_tensor.flatten()[:, None]], dim=1) # concatenate along columns
inputs3 = torch.cat([X3.flatten()[:, None], T.flatten()[:, None], q3_tensor.flatten()[:, None]], dim=1) # concatenate along columns


# Converting the observed target variable (c) into a PyTorch tensor.
c1_tensor=torch.tensor(c1, dtype=torch.float32).to(device)
c2_tensor=torch.tensor(c2, dtype=torch.float32).to(device)
c3_tensor=torch.tensor(c3, dtype=torch.float32).to(device)

# Function to generate collocation points (random points in the domain)
# def generate_collocation_points(num_points, x_start, x_end, t_start, t_end):
#     x_collocation = torch.tensor(np.random.uniform(x_start, x_end, num_points), dtype=torch.float32, requires_grad=True).to(device)
#     t_collocation = torch.tensor(np.random.uniform(t_start, t_end, num_points), dtype=torch.float32, requires_grad=True).to(device)
#     return x_collocation, t_collocation
# Function to generate collocation points (random points in the domain)
def generate_collocation_points(num_points, x_start, x_end, t_start, t_end):
    x_collocation = torch.tensor(np.random.uniform(x_start, x_end, num_points), dtype=torch.float32, requires_grad=True).to(device)
    t_collocation = torch.tensor(np.random.uniform(t_start, t_end, num_points), dtype=torch.float32, requires_grad=True).to(device)
    return x_collocation, t_collocation
# def generate_collocation_pointshq(num_points, h_start, h_end, q_start, q_end):
#     h_collocation = torch.tensor(np.random.uniform(h_start, h_end, num_points), dtype=torch.float32).to(device)
#     q_collocation = torch.tensor(np.random.uniform(q_start, q_end, num_points), dtype=torch.float32).to(device)
#     return h_collocation, q_collocation


def generate_collocation_pointsQ(num_points, Q_start, Q_end):
    Q_collocation = torch.tensor(np.random.uniform(Q_start, Q_end, num_points), dtype=torch.float32, requires_grad=True).to(device)
    return Q_collocation

def generate_collocation_pointsQ_alt(num_points, Q_start, Q_end):
    Q_collocation = np.random.uniform(Q_start, Q_end, num_points)
    return Q_collocation


#%% Training loop

epochs = config["neural_network"]["epochs"]
lambda_pde = config["neural_network"]["lambda_pde"] # Weight for PDE loss
lambda_bc = config["neural_network"]["lambda_bc"] # Weight for boundary condition loss

x1_collocation, t_collocation = generate_collocation_points(epochs, x1_start, x1_end, t_start, t_end) #10000
x2_collocation, t_collocation = generate_collocation_points(epochs, x2_start, x2_end, t_start, t_end) #10000
x3_collocation, t_collocation = generate_collocation_points(epochs, x3_start, x3_end, t_start, t_end) #10000

h1_collocation, q1_collocation = generate_collocation_points(epochs, h1_start, h1_end, q1_start, q1_end) #10000
h2_collocation, q2_collocation = generate_collocation_points(epochs, h2_start, h2_end, q2_start, q2_end) #10000
h3_collocation, q3_collocation = generate_collocation_points(epochs, h3_start, h3_end, q3_start, q3_end) #10000

# Q1_collocation= generate_collocation_pointsQ (epochs,Q1_start, Q1_end)
# Strmpowcri1_collocation = generate_collocation_pointsQ (epochs,strmpowcri1_start, strmpowcri1_end)
Q1_collocation=generate_collocation_pointsQ_alt(epochs,Q1_start, Q1_end)
Strmpowcri1_collocation = generate_collocation_pointsQ_alt (epochs,strmpowcri1_start, strmpowcri1_end)

# Q2_collocation= generate_collocation_pointsQ (epochs,Q2_start, Q2_end)
# Strmpowcri2_collocation = generate_collocation_pointsQ (epochs,strmpowcri2_start, strmpowcri2_end)

Q2_collocation= generate_collocation_pointsQ_alt (epochs,Q2_start, Q2_end)
Strmpowcri2_collocation = generate_collocation_pointsQ_alt (epochs,strmpowcri2_start, strmpowcri2_end)

# Q3_collocation= generate_collocation_pointsQ (epochs,Q3_start, Q3_end)
# Strmpowcri3_collocation = generate_collocation_pointsQ (epochs,strmpowcri3_start, strmpowcri3_end)

Q3_collocation= generate_collocation_pointsQ_alt (epochs,Q3_start, Q3_end)
Strmpowcri3_collocation = generate_collocation_pointsQ_alt (epochs,strmpowcri3_start, strmpowcri3_end)

h1_collocation_noTensor=generate_collocation_pointsQ_alt(epochs,h1_start, h1_end)
h2_collocation_noTensor=generate_collocation_pointsQ_alt(epochs,h2_start, h2_end)
h3_collocation_noTensor=generate_collocation_pointsQ_alt(epochs,h3_start, h3_end)

#%% Collocation points
x1_col= torch.tensor(x1_collocation, dtype=torch.float32).to(device)
x1_col_pd= pd.DataFrame(x1_col.cpu().numpy())
t_col= torch.tensor(t_collocation, dtype=torch.float32).to(device)
t_col_pd= pd.DataFrame(t_col.cpu().numpy())

x2_col= torch.tensor(x2_collocation, dtype=torch.float32).to(device)
x2_col_pd= pd.DataFrame(x2_col.cpu().numpy())


x3_col= torch.tensor(x3_collocation, dtype=torch.float32).to(device)
x3_col_pd= pd.DataFrame(x3_col.cpu().numpy())


q1_col= torch.tensor(q1_collocation, dtype=torch.float32).to(device)
q1_col_pd= pd.DataFrame(q1_col.cpu().numpy())

q2_col= torch.tensor(q2_collocation, dtype=torch.float32).to(device)
q2_col_pd= pd.DataFrame(q2_col.cpu().numpy())

q3_col= torch.tensor(q3_collocation, dtype=torch.float32).to(device)
q3_col_pd= pd.DataFrame(q3_col.cpu().numpy())

#%% calculate re-entrainment for collocation
Strmpow1_collocation=rho*g*Q1_collocation*slope1 # stream power
ryd1_collocation=((Fi*(Strmpow1_collocation-(Cri1*Strmpowcri1_collocation))
                /(ki*g*h1_collocation_noTensor)))*(rhos/(rhos-rho)) #re-entrainment

Strmpow2_collocation=rho*g*Q2_collocation*slope2 # stream power
ryd2_collocation=((Fi*(Strmpow2_collocation-(Cri1*Strmpowcri2_collocation))
                /(ki*g*h2_collocation_noTensor)))*(rhos/(rhos-rho)) #re-entrainment
Strmpow3_collocation=rho*g*Q3_collocation*slope3 # stream power
ryd3_collocation=((Fi*(Strmpow3_collocation-(Cri1*Strmpowcri3_collocation))
                /(ki*g*h3_collocation_noTensor)))*(rhos/(rhos-rho)) #re-entrainment

ryd1_collocation_tensor=torch.tensor(ryd1_collocation, dtype=torch.float32).to(device)
ryd2_collocation_tensor=torch.tensor(ryd2_collocation, dtype=torch.float32).to(device)
ryd3_collocation_tensor=torch.tensor(ryd3_collocation, dtype=torch.float32).to(device)
#%% output loss dataframe
columns=['epoch', 'pde_loss1','pde_loss2', 'pde_loss3','ic1_loss','ic2_loss', 'ic3_loss',
         'ic3_loss_sum', 'bc1_loss','bc2_loss','bc3_loss_sum', 'output_loss', 'Total_loss' ]
dfloss_out=pd.DataFrame(columns=columns)
#%% Mode run

min_loss = float('inf')
# epoch=1
for epoch in range(epochs):
 
    model1.train()
    model2.train()
    model3.train()
    
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()
    
    # Forward pass for training points (for boundary and initial conditions)
    predictions1 = model1(inputs1)
    c_pred1 = predictions1.reshape(x_steps, t_steps)  # Reshapes the predictions to match the dimensions of the spatial-temporal grid.
    
    predictions2 = model2(inputs2)
    c_pred2 = predictions2.reshape(x_steps, t_steps)  # Reshapes the predictions to match the dimensions of the spatial-temporal grid.
    
    predictions3 = model3(inputs3)
    c_pred3 = predictions3.reshape(x_steps, t_steps)  # Reshapes the predictions to match the dimensions of the spatial-temporal grid.
    
    
    # summing precited c from two upstream reaches
    c_pred3_sum = c_pred1 + c_pred2
    
    # Compute PDE loss
    residual1 = pde_fn(c_pred1, X1, T, h1_tensor, q1_tensor, ryd1_tensor)
    residual2 = pde_fn(c_pred2, X2, T, h2_tensor, q2_tensor, ryd2_tensor)
    residual3 = pde_fn(c_pred3, X3, T, h3_tensor, q3_tensor, ryd3_tensor)
    
    
    # Compute boundary condition loss
    ic1_loss = torch.mean((c_pred1[:, 0] - c1_tensor[:, 0])**2)
    bc1_loss = torch.mean((c_pred1[0, :] - c1_tensor[0, :])**2)
    
    ic2_loss = torch.mean((c_pred2[:, 0] - c2_tensor[:, 0])**2)
    bc2_loss = torch.mean((c_pred2[0, :] - c2_tensor[0, :])**2)
    
    ic3_loss = torch.mean((c_pred3[:, 0] - c3_tensor[:, 0])**2)
    
    # loss of intial between observed C and sum of predicted c1 and c2
    ic3_loss_sum = torch.mean((c_pred3_sum[:, 0] - c3_tensor[:, 0])**2)
    
    # loss of two predictions summing c predicted of two upstream reaches and loss of predicted c from model 
    bc3_loss_sum = torch.mean((c_pred3[0, :] - c_pred3_sum[-1, :])**2)
    
    # comparing loss between downstream c and predicted c at the last row
    output_loss = torch.mean((c_pred3[-1, :] - c3_tensor[0, :])**2)
    
    # Randomly sampled points in the  (x, t) domain
    collocation_points1 = torch.cat([x1_collocation[:, None], t_collocation[:, None], q1_collocation[:, None]], dim=1)
    collocation_points2 = torch.cat([x2_collocation[:, None], t_collocation[:, None], q2_collocation[:, None]], dim=1)
    collocation_points3 = torch.cat([x3_collocation[:, None], t_collocation[:, None], q3_collocation[:, None]], dim=1)
        
    # Model predictions at these collocation points
    collocation_pred1 = model1(collocation_points1)
    collocation_pred2 = model2(collocation_points2)
    collocation_pred3 = model3(collocation_points3)
              
    # Computing the residuals at collocation points to ensure the model satisfies the governing PDE
    collocation_residual1 = pde_fn(collocation_pred1, x1_collocation, t_collocation,h1_collocation, q1_collocation, ryd1_collocation_tensor )
    collocation_residual2 = pde_fn(collocation_pred2, x2_collocation, t_collocation,h2_collocation, q2_collocation, ryd2_collocation_tensor )
    collocation_residual3 = pde_fn(collocation_pred3, x3_collocation, t_collocation,h3_collocation, q3_collocation, ryd3_collocation_tensor )
        
    # Combining the residual loss at collocation points and grid points.
    pde_loss1 = torch.mean(collocation_residual1**2) + torch.mean(residual1**2)
    pde_loss2 = torch.mean(collocation_residual2**2) + torch.mean(residual2**2)
    pde_loss3 = torch.mean(collocation_residual3**2) + torch.mean(residual3**2)
    
    
    # revised loss
    loss= (pde_loss1 + pde_loss2 + pde_loss3 + 
           ic1_loss + bc1_loss + ic2_loss + bc2_loss + 
           ic3_loss + ic3_loss_sum + bc3_loss_sum + output_loss)
    
    # Backpropagation
    loss.backward()    # Computes gradients for all parameters in the network using backpropagation
    optimizer1.step()   # Updates the model’s parameters based on the computed gradients
    optimizer2.step()   # Updates the model’s parameters based on the computed gradients
    optimizer3.step()   # Updates the model’s parameters based on the computed gradients
    
    # Logging
    if epoch % 10 == 0:
        
        # torch.save(pde_loss1, 'pde_loss1.pth')
        # torch.save(pde_loss2, 'pde_loss2.pth')
        # torch.save(pde_loss3, 'pde_loss3.pth')
        # torch.save(ic1_loss, 'ic1_loss.pth')
        # torch.save(ic2_loss, 'ic2_loss.pth')
        # torch.save(ic3_loss, 'ic3_loss.pth')
        # torch.save(ic3_loss_sum, 'ic3_loss.pth')
        # torch.save(bc1_loss, 'bc1_loss.pth')
        # torch.save(bc2_loss, 'bc2_loss.pth')
        # torch.save(bc3_loss_sum, 'bc3_loss_sum.pth')
        # torch.save(output_loss, 'output_loss.pth')
        lossdata_out=[epoch, pde_loss1.item(), pde_loss2.item(), pde_loss3.item(), 
                      ic1_loss.item(), ic2_loss.item(), ic3_loss.item(),
                      ic3_loss_sum.item(), bc1_loss.item(), bc2_loss.item(),
                      bc3_loss_sum.item(), output_loss.item(), loss.item()]
        dfloss_out.loc[len(dfloss_out)]=lossdata_out
        print (' Error pde:\n', f'pde_loss1 = {pde_loss1.item():.2e}', 
               f'pde_loss2 = {pde_loss2.item():.2e}',
               f'pde_loss3 = {pde_loss3.item():.2e}\n',
               'Error initial condition:\n',
               f'ic1_loss = {ic1_loss.item():.2e}',
               f'ic2_loss = {ic2_loss.item():.2e}',
               f'ic3_loss = {ic3_loss.item():.2e}\n',
               'Error boundary condition:\n',
               f'bc1_loss = {bc1_loss.item():.2e}',
               f'bc2_loss = {bc2_loss.item():.2e}\n',
               'Error boundary condition of 3 for sum:\n',
               f'bc3_loss_sum = {bc3_loss_sum.item():.2e}\n',
               'Error output loss for 3:\n',
               f'output_loss_ic = {ic3_loss_sum.item():.2e}\n'
               f'output_loss = {output_loss.item():.2e}')
        print (f' Total Error = {min_loss:.2f}') 
        # print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}, PDE Loss: {pde_loss1.item():.6f}, IC Loss: {ic_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}")
    
    # Save the model whenever the total loss achieves a new minimum:
    if min_loss > loss.item():
        min_loss = loss.item()
        # save to current directory
        torch.save(model1.state_dict(), f'model1_best_loss.pth')
        torch.save(model2.state_dict(), f'model2_best_loss.pth')
        torch.save(model3.state_dict(), f'model3_best_loss.pth')
        # torch.save(model1.state_dict(), 'C:/Users/haddadchia/Desktop/model_saved/model1_best_loss.pth')
        # torch.save(model2.state_dict(), 'C:/Users/haddadchia/Desktop/model_saved/model2_best_loss.pth')
        # torch.save(model3.state_dict(), 'C:/Users/haddadchia/Desktop/model_saved/model3_best_loss.pth')
        print(f"Model saved at epoch {epoch}")


# #save loss for every  x steps as csv
dfloss_out.to_csv (os.path.join(loss_directory,output_csv_name), index=False)

#%% save the models 
# Save the model state after the final training epoch
torch.save(model1.state_dict(), f'model1_last.pth')
torch.save(model2.state_dict(), f'model2_last.pth')
torch.save(model2.state_dict(), f'model3_last.pth')
# torch.save(model1.state_dict(), 'C:/Users/haddadchia/Desktop/model1_saved/model_last.pth')
# torch.save(model2.state_dict(), 'C:/Users/haddadchia/Desktop/model2_saved/model_last.pth')
# torch.save(model3.state_dict(), 'C:/Users/haddadchia/Desktop/model3_saved/model_last.pth')
print(f"Final model saved at epoch {epoch}")




