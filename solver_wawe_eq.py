import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi

# wave speed
c = 60

# spatial domain
n = 101  # number of grid points
xmin = 0
xmax = 1

# time domain
m = 500  # number of time steps
tmin = 0
T = tmin + np.arange(m+1)
tmax = 500

# x grid of n points
X, dx = np.linspace(xmin, xmax, n, retstep=True)


dt = dx/c

# initial conditions


def initial_u(x):
    result = np.zeros(len(x), dtype='f8')
    middle_index = int((len(x))/2)
    for index in range(len(x)):
        if (index == middle_index):
            result[index] = 10
        else:
            result[index] = 1
    return result

# initial derivative


def initial_derivative(u0, dx):
    v = np.zeros(len(u0), dtype='f8')
    for i in range(0, len(u0) - 1):
        v[i] = (u0[i + 1] - u0[i]) / dx
    return v


def space_derivative(u, dx):
    derivative = np.zeros(len(u), dtype='f8')
    for i in range(1, len(u) - 1):
        derivative[i] = (u[i - 1] - 2 * u[i] + u[i + 1]) / (2 * dx * dx)
    return derivative


def derivative_2nd(u0, v0, dt, dx):
    derivative = space_derivative(u0, dx)
    const = pow(c, 2)
    k1 = derivative * const
    k2 = const * (derivative + k1 * dt / 2)
    k3 = const * (derivative + k2 * dt / 2)
    k4 = const * (derivative + k3 * dt)
    v1 = v0 + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return v1


def derivative_1st(u0, v1, dt):
    k1 = v1
    k2 = v1 + k1 * (dt / 2)
    k3 = v1 + k2 * (dt / 2)
    k4 = v1 + k3 * dt
    u1 = u0 + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return u1


# time solution
U = np.zeros((m+1, n), dtype=float)
V = np.zeros((m+1, n), dtype=float)
U[0, :] = initial_u(X)
V[0, :] = initial_derivative(U[0, :], dx)
u = U[0, :]
v = V[0, :]

for k in range(m):
    V[k + 1, :] = derivative_2nd(U[k, :], V[k, :], dt, dx)
    U[k + 1, :] = derivative_1st(U[k, :], V[k + 1, :], dt)

    print(U[k, 23:27])


# plot solution
plt.style.use('dark_background')
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# animate the time data
k = 0


def animate(i):
    global k
    x = U[k, :]
    k += 1
    ax1.clear()
    plt.plot(X, x, color='cyan')
    plt.grid(True)
    plt.ylim([-10, 10])
    plt.xlim([xmin, xmax])


anim = animation.FuncAnimation(fig, animate, frames=360, interval=20)
plt.show()
