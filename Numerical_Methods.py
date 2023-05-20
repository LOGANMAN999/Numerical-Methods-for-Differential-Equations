import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logan Singerman
# 5/1/2023
# Numerical Methods for Differential Equations 

"""
Required Libraries:

This Python script relies on the following libraries:
- numpy
- pandas
- matplotlib

Please make sure these libraries are installed before running the script.
If you don't have them installed, you can install them using pip:

1. Open a terminal (command prompt on Windows).
2. Run the following command to install the libraries:

   pip install numpy pandas matplotlib

If you're using a Python virtual environment, activate the virtual environment first before running the above command.

After installing the libraries, you should be able to run this script without issues.
"""





# Exact solution and derivatives used in methods
def function_1_exact(t):
    return np.cos(t) + np.sin(t) + (1/3)*t**3 + 2

def function_1_prime(t):
    return np.cos(t) - np.sin(t) + t**2

def function_1_prime2(t):
    return -np.sin(t) - np.cos(t) + 2*t

def function_1_prime3(t):
    return -np.cos(t) + np.sin(t) + 2

def function_1_prime4(t):
    return np.sin(t) + np.cos(t)

# TS(3) method

def TS3(x0, t0, h):
    ts3_graph = []
    x = x0
    t = t0
    n = 1/h
    for i in range(int(n+1)):
        ts3_graph.append(x)
        x = x + h*(function_1_prime(t)) + (0.5*h**2)*(function_1_prime2(t)) + (1/6*h**3)*(function_1_prime3(t))
        t += h
    return ts3_graph

# TS(4) method

def TS4(x0, t0, h):
    ts4_graph = []
    x = x0
    t = t0
    n = 1/h
    for i in range(int(n+1)):
        ts4_graph.append(x)
        x = x + h*(function_1_prime(t)) + (0.5) * (h**2)*(function_1_prime2(t)) + (1/6) * (h**3)*(function_1_prime3(t)) + (1/24) * (h**4)*(function_1_prime4(t))
        t += h
    return ts4_graph

# initial conditons and 
# initialization for Question 1

q1_x0 = 3
t0 = 0
h1 = [0.2, 0.1, 0.05]
xTS3= []
xTS4 = []
GE_TS3 = []
GE_TS4 = []
GE_TS3_C = []
GE_TS4_C = []
t_values_1 = np.linspace(0, 1, 35)
# computing approximation values
# with different step sizes

plt.figure()

for i in h1:
    sol1 = TS3(q1_x0, t0, i)
    plt.plot(np.linspace(0, 1, int(1/i)+1), sol1, label="TS3 h="+ str(i), linestyle="dashed", linewidth=2)
    xTS3.append(sol1[-1])
    GE_TS3.append(function_1_exact(1)-sol1[-1])
    GE_TS3_C.append((function_1_exact(1)-sol1[-1])/i**3)

    sol2 = TS4(q1_x0, t0, i)
    plt.plot(np.linspace(0, 1, int(1/i)+1), sol2, label="TS4 h="+ str(i), linestyle="dotted", linewidth=2)
    xTS4.append(sol2[-1])
    GE_TS4.append(function_1_exact(1)-sol2[-1])
    GE_TS4_C.append((function_1_exact(1)-sol2[-1])/i**4)

# Data frame of method outputs
data_TS = pd.DataFrame({
    "h": h1,
    "TS3": xTS3,
    "TS4": xTS4,
    "GE-TS3": GE_TS3,
    "GE-TS4": GE_TS4,
    "GE-TS3 /h^3": GE_TS3_C,
    "GE-TS4 /h^4": GE_TS4_C})
print(data_TS)


exact_graph_1 = [function_1_exact(t) for t in t_values_1]

plt.plot(t_values_1, exact_graph_1, label="Exact Solution", linestyle="solid", linewidth=2)

plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("Exact Solution, TS3, and TS4")
plt.legend()
#plt.show()



def function_2_prime(t, x):
    return -20 * x

def RK4(f, t0, x0, h):
    RK4_graph = []
    t = t0
    x = x0
    RK4_graph.append(x)
    n = 2/h
    for i in range(int(n+1)):
        k1 = f(t, x)
        k2 = f(t + h/2, x + h/2 * k1)
        k3 = f(t + h/2, x + h/2 * k2)
        k4 = f(t + h, x + h * k3)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        RK4_graph.append(x)
        t += h
    return RK4_graph

def RK3(f, t0, x0, h):
    RK3_graph = []
    t = t0
    x = x0
    RK3_graph.append(x)
    n = 2/h
    for i in range(int(n+1)):
        k1 = f(t, x)
        k2 = f(t + h/2, x + h/2 * k1)
        k3 = f(t + h, x - h*k1 + 2*h*k2)
        x = x + h/6 * (k1 + 4*k2 + k3)
        RK3_graph.append(x)
        t += h
    return RK3_graph
    
def function_2_exact(t):
    return np.exp(-20*t)

q2_x0 = 1
h2 = [1/4, 1/8, 1/16]
xRK4 = []
xRK3 = []
GE_RK3 = []
GE_RK4 = []
t_values_2 = np.linspace(0, 2, 100)
plt.figure()

for i in h2:
    RK4_sol = RK4(function_2_prime, t0, q2_x0, i)
    plt.plot(np.linspace(0, 2, int(2/i)+2), RK4_sol, label="RK4 h="+ str(i), linestyle="dashed", linewidth=2)
    xRK4.append(RK4_sol[-1])
    GE_RK4.append(function_2_exact(2)-RK4_sol[-1])
    RK3_sol = RK3(function_2_prime, t0, q2_x0, i)
    plt.plot(np.linspace(0, 2, int(2/i)+2), RK3_sol, label="RK3 h="+ str(i), linestyle="dashed", linewidth=2)
    xRK3.append(RK3_sol[-1])
    GE_RK3.append(function_2_exact(2)-RK3_sol[-1])

data_RK = pd.DataFrame({
    "h": h2,
    "RK3 t=2": xRK3,
    "RK4 t=2": xRK4,
    "GE_RK3 t=2": GE_RK3,
    "GE_RK4 t=2": GE_RK4
})

exact_graph_2 = [function_2_exact(t) for t in t_values_2]
plt.plot(t_values_2, exact_graph_2, label="Exact Solution", linestyle="solid", linewidth=2)

plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("Exact Solution, RK3, and RK4")
plt.legend()
#plt.show()
print(data_RK)


# plot the region of absolute stability for RK4
def stability_function_RK4(z):
     return 1 + z + 1/2 * z**2 + 1/6 * z**3 + 1/24 * z**4


stability_plot_RK4x = np.linspace(-5, 5, 500)
stability_plot_RK4y = np.linspace(-5, 5, 500)
X, Y = np.meshgrid(stability_plot_RK4x, stability_plot_RK4y)
Z = X + 1j * Y

# Compute the absolute value of the stability function
R = np.abs(stability_function_RK4(Z))

# Plot the region of absolute stability
plt.figure(figsize=(6, 6))
plt.contour(X, Y, R, levels=[1], colors='blue')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')
plt.title('Region of Absolute Stability of RK4')
plt.grid()
plt.axis('equal')


# plot the region of absolute stability of RK3
def stability_function_RK3(z):
    return 1 + z + 1/2 * z**2 + 1/6 * z**3


stability_plot_RK3x = np.linspace(-5, 5, 500)
stability_plot_RK3y = np.linspace(-5, 5, 500)
W, U = np.meshgrid(stability_plot_RK3x, stability_plot_RK3y)
Z2 = W + 1j * U

# Compute the absolute value of the stability function
R2 = np.abs(stability_function_RK3(Z2))

# Plot the region of absolute stability
plt.figure(figsize=(6, 6))
plt.contour(W, U, R2, levels=[1], colors='blue')
plt.xlabel('Re(hλ)')
plt.ylabel('Im(hλ)')
plt.title('Region of Absolute Stability of RK3')
plt.grid()
plt.axis('equal')
#plt.show()



def function_3_exact(t):
    return t - (np.exp(-40)-np.exp(-40*(1-t)))/1-np.exp(-40)

def finite_difference(h):
   N = int(1/h) - 1
   x = np.linspace(0, 1, N+2)
   A = np.zeros((N, N))
   b = np.full(N, 2)

   b[0] = 0
   b[-1] = 2 

   for i in range(N):
       A[i, i] = 0.1/h**2
       
       if i > 0:
           A[i, i-1] = ((-0.05-h)/h**2)
       if i < N-1:
           A[i, i+1] = ((-0.05+h)/h**2)
   U = np.linalg.solve(A, b)

   x_sol = np.concatenate(([0], U, [2]))
   return x, x_sol 
	
h3 = [1/10, 1/19, 1/21, 1/42]
t_values_3 = np.linspace(0, 1, 100)
max_differences = []
plt.figure()
for h in h3:
    BVP_domain, BVP_sol = finite_difference(h)
    differences = [(np.abs(function_3_exact(t)-BVP_sol[i])) for t,i in zip(BVP_domain, range(len(BVP_domain)))]
    max_diff = np.max(differences)
    max_differences.append(max_diff)
    plt.plot(BVP_domain, BVP_sol, label="Finite Difference h="+ str(h), linestyle="dashed", linewidth=2)

exact_graph_3 = [function_3_exact(t) for t in t_values_3]
plt.plot(t_values_3, exact_graph_3, label="Exact Solution", linestyle="solid", linewidth=2)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title("Exact Solution and Finite Difference")
plt.legend()
#plt.show()

data_FD = pd.DataFrame({"h": h3, "Max Difference": max_differences})
print(data_FD)



ub = lambda x: np.sin(np.pi * x) - np.sin(3 * np.pi * x)
uleft = lambda t: 0
uright = lambda t: 0

# Exact solution
u_exact = lambda x, t: np.exp(-np.pi**2 * t) * np.sin(np.pi * x) - np.exp(-9 * np.pi**2 * t) * np.sin(3 * np.pi * x)

# Parameters (r=k/h^2)

# FTCS
error_FTCS = []
r_vals = [1/6, 1/3, 1, 10]
for i in r_vals:
	M = 10
	h = 1 / M
	r = i
	k = r * h**2
	tau = 0.2
	x = np.linspace(0, 1, M + 1)
	xi = x[1:-1]
	t = np.arange(0, tau + k, k)

	# Set up matrices
	D = 2 * np.eye(M - 1) - np.diag(np.ones(M - 2), 1) - np.diag(np.ones(M - 2), -1)
	I = np.eye(M - 1)
	A = I - r * D

	# U0
	uin = ub(xi)
	u0 = uin
	usol = np.zeros((len(xi), len(t)))
	uex = np.zeros((len(xi), len(t)))
	uex[:, 0] = u0
	usol[:, 0] = u0

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	fig.suptitle(f"FTCS Approximation h = 1/10, k = {k:.{5}}", fontsize = 20)
	for ii in range(1, len(t)):
		u1 = A @ u0
		u0 = u1
		usol[:, ii] = u1
		uex[:, ii] = u_exact(xi, t[ii])
		
		if ii % 2 == 0:
			ax.plot(xi, t[ii] * np.ones(M - 1), u1, 'k')
			
	error = np.max(np.abs(usol[:, -1] - uex[:, -1]))
	error_FTCS.append(error)
	#print(f'Error FTCS ={error}')

# BTSC
error_BTSC = []
for i in r_vals:
    M = 10
    h = 1 / M
    r = i
    k = r * h**2
    tau = 0.2
    x = np.linspace(0, 1, M + 1)
    xi = x[1:-1]
    t = np.arange(0, tau + k, k)

    # Set up matrices
    A = (1 + 2 * r) * np.eye(M - 1) - r * np.diag(np.ones(M - 2), 1) - r * np.diag(np.ones(M - 2), -1)
    A_inv = np.linalg.inv(A)

    # U0
    uin = ub(xi)
    u0 = uin
    usol = np.zeros((len(xi), len(t)))
    uex = np.zeros((len(xi), len(t)))
    uex[:, 0] = u0
    usol[:, 0] = u0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.suptitle(f"BTCS Approximation h = 1/10, k = {k:.{5}}", fontsize = 20)
    for ii in range(1, len(t)):
        u1 = A_inv @ u0
        u0 = u1
        usol[:, ii] = u1
        uex[:, ii] = u_exact(xi, t[ii])

        if ii % 2 == 0:
            ax.plot(xi, t[ii] * np.ones(M - 1), u1, 'k')

    error = np.max(np.abs(usol[:, -1] - uex[:, -1]))
    error_BTSC.append(error)
    #print(f'Error BTSC ={error}')

# Crank-Nicholson
error_cn = []
for i in r_vals:
    M = 10
    h = 1 / M
    r = i
    k = r * h**2
    tau = 0.2
    x = np.linspace(0, 1, M + 1)
    xi = x[1:-1]
    t = np.arange(0, tau + k, k)

    # Set up matrices
    A = (1 + r) * np.eye(M - 1) - 0.5 * r * np.diag(np.ones(M - 2), 1) - 0.5 * r * np.diag(np.ones(M - 2), -1)
    B = (1 - r) * np.eye(M - 1) + 0.5 * r * np.diag(np.ones(M - 2), 1) + 0.5 * r * np.diag(np.ones(M - 2), -1)
    
    # U0
    uin = ub(xi)
    u0 = uin
    usol = np.zeros((len(xi), len(t)))
    uex = np.zeros((len(xi), len(t)))
    uex[:, 0] = u0
    usol[:, 0] = u0

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.suptitle(f"Crank-Nicolson Approximation h = 1/10, k = {k:.{5}}", fontsize = 20)
    for ii in range(1, len(t)):
        u1 = np.linalg.solve(A, B @ u0)
        u0 = u1
        usol[:, ii] = u1
        uex[:, ii] = u_exact(xi, t[ii])

        if ii % 2 == 0:
            ax.plot(xi, t[ii] * np.ones(M - 1), u1, 'k')

    error = np.max(np.abs(usol[:, -1] - uex[:, -1]))
    error_cn.append(error)
    #print(f'Error Crank-Nicholson={error}')

data_PDE = pd.DataFrame({"r": r_vals,
                         "Error FTSC": error_FTCS,
                         "Error BTSC": error_BTSC,
                         "Error Crank-Nicholson": error_cn})
print(data_PDE)

plt.show()
