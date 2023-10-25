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
        self.n_layers = 20
        self.hidden_dim = 128

        self.rnn = nn.RNN(12, self.hidden_dim, self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, encode_layers[-1])

        self.l2_regularization = nn.Parameter(torch.tensor(0.001, requires_grad=False))
        self.dropout = nn.Dropout(0.20)

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
    error_prediction_real_new = error_prediction_real
    
    loss =  0.0*torch.norm(error_koop_new, p=2) + 1*torch.norm(error_prediction_real_new, p='fro')
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.A.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

def learning_network(neural_network, num_epochs, lr, Kbatch_size):
    # Optimizaer definition
    optimizer = torch.optim.Adam(neural_network.parameters(), lr)

    # Epochs parameters
    losses = defaultdict(lambda: defaultdict(list))
    #Kbatch_size = 200

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
    return neural_network, losses

    
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

neural_network, losses = learning_network(neural_network, 10000, 0.001, 500)

# Validation Data = scipy.io.loadmat('blue_data_02.mat')
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

h, hp, T = get_odometry(data_odom_blue, steering_real, vx, vy, vz, wx, wy, wz, vel_control, steering_control, 0, 500)
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

A_a = neural_network.A.weight.cpu()
A_a = A_a.double()
A_a = A_a.detach().numpy()

B_a = neural_network.B.weight.cpu()
B_a = B_a.double()
B_a = B_a.detach().numpy()

C_ones = np.eye(X1.shape[0], dtype = np.double)
C_zeros = np.zeros((X1.shape[0], n - X1.shape[0]), dtype=np.double)
C_a = np.hstack((C_ones, C_zeros))

## Plot matrix A
plt.imshow(A_a)
plt.colorbar()
plt.show()
#
# Plot matrix B
plt.imshow(B_a)
plt.colorbar()
plt.show()
#
# New variables in order to verify the identification
x_estimate = np.zeros((X1.shape[0], X1.shape[1]+1), dtype=np.double)
output_estimate = np.zeros((X1.shape[0], U.shape[1]), dtype=np.double)
output_real = np.zeros((X1.shape[0], U.shape[1]), dtype=np.double)
error_vector = np.zeros((X1.shape[0], U.shape[1]), dtype=np.double)
norm_error = np.zeros((1, U.shape[1]), dtype = np.double)


# Tensors of the system
X1_tensor =  torch.tensor(X1).to(device)
X1_tensor = X1_tensor.double()

X2_tensor =  torch.tensor(X2).to(device)
X2_tensor = X2_tensor.double()

aux_init = X1_tensor.T
aux_init = aux_init.unsqueeze(0)
x_aux_estimate, _ = neural_network.encode(aux_init)
x_aux_estimate = x_aux_estimate.cpu().double().detach().numpy()
# Initial value
x_estimate[:, 0] = C_a@x_aux_estimate[0, 0, :]

for k in range(0, U.shape[1]):
    output_estimate[:, k] = x_estimate[:, k]
    
    output_numpy_pytorch = torch.from_numpy(X1[:, k]).to(device).double()
    output_numpy_pytorch = torch.reshape(output_numpy_pytorch, (1, 1, X1_tensor.shape[0]))
    output_aux, _ = neural_network.encode(output_numpy_pytorch)
    
    output_aux = output_aux.cpu().double().detach().numpy()
    output_real[:, k] = C_a@output_aux[0, 0, :]

    error_vector[:, k] = output_real[:, k] - output_estimate[:, k]
    norm_error[:, k] = np.linalg.norm(error_vector[:, k])
    
    # transformation between pytorch an numpy
    aux_numpy_pytorch = torch.from_numpy(x_estimate[:, k]).to(device).double()
    aux_numpy_pytorch = torch.reshape(aux_numpy_pytorch, (1, 1, X1_tensor.shape[0]))

    x_aux_estimate, _ = neural_network.encode(aux_numpy_pytorch)
    x_aux_estimate = x_aux_estimate.cpu().double().detach().numpy()
    
    aux_states = x_aux_estimate[0, 0, :]
    x_estimate[:, k+1] = C_a@(A_a@aux_states + B_a@U[:, k])

print("Error estimation norm")
print(np.linalg.norm(norm_error))
eig_A, eigv_A = np.linalg.eig(A_a)
print("Print Eigvalues A")
print(eig_A)

fig13, ax13, ax23, ax33 = fancy_plots_3()
plot_states_angles_estimation(fig13, ax13, ax23, ax33, h[7:10, :], output_estimate[:, :], t, "Euler Angles Of the system")
plt.show()

fig15, ax15, ax25, ax35 = fancy_plots_3()
plot_states_velocity_lineal_estimation(fig15, ax15, ax25, ax35, hp[0:3, :], output_estimate[:, :], t, "Lineal Velocity of the system")
plt.show()

fig16, ax16, ax26, ax36 = fancy_plots_3()
plot_states_velocity_angular_estimation(fig16, ax16, ax26, ax36, hp[3:6, :], output_estimate[:, :], t, "Angular Velocity of the system")
plt.show()

fig17, ax17, ax27 = fancy_plots_2()
plot_control_states_estimation(fig17, ax17, ax27, h[:, :], hp[:, :], output_estimate[:, :], t, "Control and Real Values of the system")
plt.show()

fig14, ax14, ax24, ax34 = fancy_plots_3()
plot_states_position_estimation(fig14, ax14, ax24, ax34, h[0:3, :], output_estimate[:, :], t, "Position of the system")
plt.show()

fig18, ax18 = fancy_plots_1()
plot_error_estimation(fig18, ax18, norm_error, t, 'Error Norm of the Estimation')
plt.show()

losses["Koopman"]["collocation"]
costo = np.array(losses["Koopman"]["collocation"])
epochs = np.array(losses["Koopman"]["num_epochs"])
costo = costo.reshape(1, costo.shape[0])

fig19, ax19 = fancy_plots_1()
plot_error_estimation(fig19, ax19, costo, epochs, 'Training Cost')
plt.show()