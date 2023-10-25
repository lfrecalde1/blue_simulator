import matplotlib
matplotlib.use('TkAgg')
import scipy.io
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from fancy_plots import fancy_plots_3
from fancy_plots import plot_states_angles_estimation, plot_states_velocity_lineal_estimation, plot_states_velocity_angular_estimation, plot_control_states_estimation
from fancy_plots import fancy_plots_2
from fancy_plots import fancy_plots_1, plot_error_estimation, plot_states_position_estimation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.nn.modules.container import T
from tqdm import tqdm
from collections import defaultdict
from collections import OrderedDict
import random

# Check Gpu
if torch.cuda.is_available(): 
    dev = "cuda:0" 
else: 
    dev = "cpu"
device = torch.device(dev)
print(torch.cuda.is_available())

# Load Data from .mat file
def get_odometry(data, angle, vx, vy, vz, wx, wy, wz, vel_control, steer_control, samples_i, samples_f):
    # Get size of the data
    i, j = data.shape
    # Init empty values
    x = np.zeros((1, j), dtype = np.double)
    y = np.zeros((1, j), dtype = np.double)
    z = np.zeros((1, j), dtype = np.double)
    quatenions = np.zeros((4, j), dtype = np.double)
    orientation_aux = np.zeros((3, j), dtype = np.double)
    
    for k in range(0, j):
        
        # Get Position
        x[:, k] = data[0, k]
        y[:, k] = data[1, k]
        z[:, k] = 0
        
        # Get quaternion
        quatenions[:, k] = [data[2, k], data[3, k], data[4, k], data[5, k]]
        
        # Get Euler Angles
        aux =  quatenions[:, k]
        r = R.from_quat(aux)
        orientation_aux[:, k] = r.as_euler('xyz', degrees = False)
        
    # get complete data of the system
    h = np.array([x[0,:], y[0,:], z[0,:],
                quatenions[0, :], quatenions[1, :], quatenions[2, :], quatenions[3, :],
                orientation_aux[0, :], orientation_aux[1, :], orientation_aux[2, :],
                angle[0, :]], dtype =np.double) 
    
    # Get Velocities of the system
    hp = np.array([vx[0, :], vy[0, :], vz[0, :], wx[0, :], wy[0, :], wz[0, :]], dtype = np.double)
    T = np.array([vel_control[0,:], steer_control[0, :]], dtype = np.double)
    return h[:, samples_i:samples_f+1], hp[:, samples_i:samples_f+1], T[:, samples_i:samples_f]

# Get X1,X2,U matrices from the system
def get_simple_data(h, hp, T):
    ## Position
    x = h[0, :]
    y = h[1, :]
    ## Linear velocities
    vx = hp[0, :]
    vy = hp[1, :]
    vz = hp[2, :]
    
    ## Get angular velocities
    p = hp[3, :]
    q = hp[4, :]
    r = hp[5, :]
    
    ## Angular velocities vector
    omega = hp[3:6, :]
    
    ## Orientacion
    quaternion = h[3:7, :]
    
    ##euler
    euler = h[7:10, :]
    
    ## Steer angle = 
    alpha = h[10, :]
    
    ## General states data
    #X = np.array([euler[2,:], omega[2, :], alpha, vx, vy], dtype = np.double)
    #X = np.array([euler[0, :], euler[1, :], euler[2, :], omega[0, :], omega[1, :], omega[2, :], alpha, vx, vy, x, y], dtype = np.double)
    X = np.array([euler[2, :], omega[2, :], alpha, vx, x, y], dtype = np.double)
    ## Control Action
    U_ref = T[:, :]
    
    ## Get the dimension of the Data
    i, j = X.shape
    
    X1 = X[:, 0:j-1]
    X2 = X[:, 1:j]
    return X1, X2, U_ref

# Lift Space Koopman Operator Matricial Form
def liftFun(x):
    x_lift = []
    for k in x: x_lift.append(k)
    x_lift.append(np.tan(x[2, :]))

    x_lift.append(np.tan(x[2, :])*x[3, :])

    x_lift.append(np.cos(x[2, :])*x[3, :])
    x_lift.append(np.sin(x[2, :])*x[3, :])
    
    x_lift.append(np.cos(x[0, :])*x[3, :])

    x_lift.append(np.sin(x[0, :])*x[3, :])

    x_lift = np.array(x_lift, dtype = np.double)
    return x_lift

# Lift Space Koopman vector form
def liftFun_vector(x):
    x_lift = []
    for k in x: x_lift.append(k)
    x_lift.append(np.tan(x[2]))

    x_lift.append(np.tan(x[2])*x[3])

    x_lift.append(np.cos(x[2])*x[3])
    x_lift.append(np.sin(x[2])*x[3])
    
    x_lift.append(np.cos(x[0])*x[3])

    x_lift.append(np.sin(x[0])*x[3])

    x_lift = np.array(x_lift, dtype = np.double)
    return x_lift

# Neural Network
class koop_model(torch.nn.Module):
    def __init__(self, encode_layers, n, m, n_normal, real_output):
        super(koop_model,self).__init__()
        Layers = OrderedDict()
        self.n_layers = 6
        self.hidden_dim = 28
        aux = 0
        #Layers["dropout_{}".format(0)] = nn.Dropout(0.20)
        #Layers["RNN_{}".format(1)] = nn.RNN(12, self.hidden_dim, self.n_layers, batch_first=True) 
        #Layers["Linear_{}".format(2)] = nn.Linear(self.hidden_dim, encode_layers[-1])
        
        self.rnn = nn.RNN(12, self.hidden_dim, self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, encode_layers[-1])
    
        self.A = nn.Linear(n, n,bias=False)
        self.A.weight.data = torch.eye(n)*0.1
        self.B = nn.Linear(m, n,bias=False)
        self.B.weight.data = torch.ones((n, m)) 
        
        self.C_eye = torch.eye(n_normal, device=device)
        self.C_zeros = torch.zeros((n_normal, n - n_normal), device=device)
        self.C= torch.cat((self.C_eye, self.C_zeros), dim=1)
        self.C = self.C.double()
        
        
        self.C_eye_1 = torch.eye(real_output, device=device)
        self.C_zeros_1 = torch.zeros((real_output, n -real_output), device=device)
        self.C_1= torch.cat((self.C_eye_1, self.C_zeros_1), dim=1)
        self.C_1 = self.C_1.double()
        

    def encode(self,x, hidden=None):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        if hidden is None:
            hidden = self.init_hidden(batch_size)
            
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return torch.cat([x,out],axis=-1), hidden
    
    def forward(self, X1, U):
        Gamma_1 = X1
        
        output_1 = self.A(Gamma_1) + self.B(U)
        output_2 = (self.A(Gamma_1) + self.B(U))@self.C.T
        output_3 = (self.A(Gamma_1) + self.B(U))@self.C_1.T
        
        return output_1, output_2, output_3
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device).double()
        return hidden

# Cost function definition
def cost_koopman(X1, X2, U, net):
    X2 = X2.T
    X1 = X1.T
    
    X2 = X2.unsqueeze(0)
    X1 = X1.unsqueeze(0)
    
    x_k, hidden = net.encode(X2)
    x_k = x_k@net.C.T
    
    x_k_real, hidden = net.encode(X2)
    x_k_real = x_k_real@net.C_1.T
    
    Gamma_k, hidden = net.encode(X2)
    
    X1, hidden = net.encode(X1)
    U = U.T
    
    # output Neural Network
    output_koopman, output_prediction, output_prediction_real = net.forward(X1, U)
    
    # Get Error
    error_koop = Gamma_k - output_koopman
    error_prediction = x_k - output_prediction
    error_prediction_real = x_k_real - output_prediction_real
    
    
    error_koop_new = error_koop.reshape((error_koop.shape[1]*error_koop.shape[2], 1))
    error_prediction_new = error_prediction.reshape((error_prediction.shape[1]*error_prediction.shape[2], 1))
    error_prediction_real_new = error_prediction_real.reshape((error_prediction_real.shape[1]*error_prediction_real.shape[2], 1))
    
    loss =  0.1*torch.norm(error_koop_new, p=2) + 1*torch.norm(error_prediction_real_new, p=2)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.A.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

## Load Matrices from mat file
Data = scipy.io.loadmat('blue_data_02.mat')

## Get odometry of the system
data_odom_blue = Data['data_odom_blue']
data_odom_blue = data_odom_blue.T

## Get Control steer angle
steering_control = Data['steering_control']
steering_control = steering_control.T
steering_control = steering_control*(np.pi/180)

## Get Steer angle real
steering_real = Data['steering_real']
steering_real = steering_real.T
steering_real = steering_real*(np.pi/180)

## Get system velocities
vx = Data['vel_real']
vx = vx.T
vy = Data['vy']
vy = vy.T
vz = Data['vz']
vz = vz.T
wx = Data['wx']
wx = wx.T
wy = Data['wy']
wy = wy.T
wz = Data['wz']
wz = wz.T

## Get desired frontal velocity
vel_control = Data['vel_control']
vel_control = vel_control.T

h, hp, T = get_odometry(data_odom_blue, steering_real, vx, vy, vz, wx, wy, wz, vel_control, steering_control, 0, 1000)

## Compute sample time of the system
ts = 0.05
t = np.zeros((T.shape[1]), dtype = np.double)
for k in range(0, T.shape[1]-1):
    t[k+1] = t[k] + ts

## Get Data DMD
X1_n, X2_n, U_n = get_simple_data(h, hp, T)
n_normal = X1_n.shape[0]

# Koopman Space
X1 = liftFun(X1_n)
X2 = liftFun(X2_n)
U = U_n

# get dimension New Space
n = X1.shape[0]
m = U.shape[0]

## Load Data in Pytorch
X1_tensor =  torch.tensor(X1,  requires_grad = True).to(device)
X1_tensor = X1_tensor.double()

X2_tensor =  torch.tensor(X2).to(device)
X2_tensor = X2_tensor.double()


C_eye = torch.eye(n_normal,  device=device)
C_zeros = torch.zeros((n_normal, n - n_normal), device=device)
C = torch.cat((C_eye, C_zeros), dim=1)

U_tensor =  torch.tensor(U, requires_grad = True).to(device)
U_tensor =  U_tensor.double()

# Neural Network Parameters
encode_dim = 30
layer_depth = 3
layer_width = 128
layers = [n] + [layer_width]*layer_depth+[encode_dim]
n = X1_tensor.shape[0] + encode_dim
neural_network = koop_model(layers, n, m, X1_tensor.shape[0], n_normal)
if torch.cuda.is_available():
    print("Yes")
    neural_network.cuda() 
neural_network.double()

# Optimizer Parameters
optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.0005)

# Epochs parameters
losses = defaultdict(lambda: defaultdict(list))
num_epochs = 10000
Kbatch_size = 200

for epoch in tqdm(range(num_epochs), desc="Koopman Neural Network: training epoch"):
        #loss.backward(retain_graph = True)
        Kindex = list(range(X1_tensor.shape[1]))
        random.shuffle(Kindex)
        
        
        Kloss = cost_koopman(X1_tensor[:, Kindex[:Kbatch_size]], X2_tensor[:, Kindex[:Kbatch_size]], U_tensor[:, Kindex[:Kbatch_size]], neural_network)
        Eloss =  Eig_loss(neural_network)
        loss = Kloss

        # Optimize Network
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
        
        
        losses["Koopman"]["collocation"].append(loss.item())
        losses["Koopman"]["num_epochs"].append(epoch)



