"""
	This program solves the drift/diffusion equation
                phi_xx = n-p
		n_t = n_xx-n*phi_xx-n_x*phi_x
                p_t =-p_xx-p*phi_xx-p_x*phi_x 
	with dirichlet boundary condition for phi
		phi(0,t) = 1; phi(1,t) = 0
        zero flux for n, p at the boundaries:
                n*phi_x - n_x = 0
                p*phi_x + p_x = 0
	with the Initial Conditions
		n(x,0) = 1
	over the domain x = [0, 1]
"""
 
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0 
# Number of internal points
N = 2000
 
# Calculate Spatial Step-Size
h = 1/(N+1.0)
 
# Create Temporal Step-Size, TFinal, Number of Time-Steps
k = h/2
TFinal = 1
NumOfTimeSteps = int(TFinal/k)
 
# Create grid-points on x axis
x = np.linspace(0,1,N+2)
x = x[1:-1]
 
# Initial Conditions
phi = np.transpose(np.mat(np.linspace(0,1,N)))
u = np.transpose(np.mat(np.linspace(1,1,N)))
v = np.transpose(np.mat(np.linspace(1,1,N))) 
phi_old = np.transpose(np.mat(np.linspace(0,1,N)))
u_old = np.transpose(np.mat(np.linspace(1,1,N)))
v_old = np.transpose(np.mat(np.linspace(1,1,N)))

# Second-Derivative Matrix
phi_x=np.gradient(x,x)
data = np.ones((3, N))
data2= np.ones((3, N))
data[1] = -2*data[1]
data2[2] = -1
data2[1] = 0
diags = [-1,0,1]
D2 = sparse.spdiags(data,diags,N,N)/(h**2)
D  = sparse.spdiags(data2,diags,N,N)/2/h
# Identity Matrix
I = sparse.identity(N)
# indentity vector
Iv = (np.ones((N,1)))
# Data for each time-step
data = []
for i in range(256): #NumOfTimeSteps):
    for j in range(4): # in each time step, update lamda and solution until the constrain is satisfied.
    #solving without constrain 
        #---------------------solve n ---------------------
        #Solve the System: (I - k/2*D2) u_new = (I + k/2*D2)*u_old
        A = (I -k/2*D2)
        b = ( I + k/2*D2 )*u_old  - k*np.multiply((D*u_old),(D*phi)) -k*np.multiply(u_old,D2*phi)
        #nx = np.gradient(u)
        #left boundary
        A[0,0] = -1./h
        A[0,1] = 1./h
        b[0] = u[0]*(phi[1]-phi[0])/h
        #right boundary
        A[N-1,N-1] = 1./h
        A[N-1,N-2] = -1./h
        b[N-1] = u[N-1]*(phi[N-1]-phi[N-2])/h
        u = np.transpose(np.mat( sparse.linalg.spsolve( A,  b ) ))
        #############adding constrain
        lamda = np.transpose(A)*(A*u-b)*2
        c = np.multiply(lamda,(u-11*Iv)) 
        c = np.multiply(c,(A*u-b))
        u = np.transpose(np.mat( sparse.linalg.spsolve( A,  c+b ) ))
        #----------------------solve p----------------------
        A = (I - k/2*D2)
        b = ( I + k/2*D2 )*v_old + k*np.multiply((D*v_old),(D*phi)) + k*np.multiply(v_old,D2*phi)
        #nx = np.gradient(u)
        #left boundary
        A[0,0] = 1./h
        A[0,1] = -1./h
        b[0] = v[0]*(phi[1]-phi[0])/h
        #right boundary
        A[N-1,N-1] = -1./h
        A[N-1,N-2] = 1./h
        b[N-1] = v[N-1]*(phi[N-1]-phi[N-2])/h
        v = np.transpose(np.mat( sparse.linalg.spsolve( A,  b ) ))

        #---------------------solve phi---------------------
        A_phi = I+D2
        b_phi = 1000*I*(u-v)
        A_phi[0,0] = 2
        A_phi[0,1] = 0
        A_phi[N-1,N-1] = 2
        A_phi[N-1,N-2] = 0
        A_phi = A_phi-I
        b_phi[0] = 0
        b_phi[N-1] = 1
        phi = np.transpose(np.mat( sparse.linalg.spsolve( A_phi,  b_phi , use_umfpack=True) ))
	# Save Data
	#data.append(u)
    #solve for lambda 
       # lamda = np.transpose(A)*(A*u-b)*2
        
       # c = np.multiply(lamda,(u-11*Iv))
       # print lamda, (u)
       # c = np.multiply(c,(A*u-b))
    #solve the constrained problem
       # u = np.transpose(np.mat( sparse.linalg.spsolve( A,  c+b ) ))
            
    u_old = u
    v_old = v
    phi_old = phi
    data.append(u)
    data.append(v)
    data.append(phi)


# Function to plot any given Frame
def plotFunction():
	plt.plot(x, data[293])
        plt.plot(x, data[294])
        plt.plot(x, data[295])
	plt.axis((0,1,0,2.1))
        
        plt.show()

# plot last step
plotFunction()
text = raw_input("prompt")  # Python 2 
