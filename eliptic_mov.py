import numpy as np
import matplotlib.pyplot as plt

# función y gradiente
def f(x1, x2, gamma):
    return 0.5*(x1**2 + gamma*x2**2)

#aqui el gradiente es el vector de derivadas parciales
def grad(x, gamma):
    x1, x2 = x
    return np.array([x1, gamma*x2], dtype=float)

# gradient descent
def gd_path(x0, alpha, gamma, max_iter=50, eps=1e-6):
    x = np.array(x0, dtype=float)
    path = [x.copy()]

    for k in range(max_iter):
        g = grad(x, gamma)
        gnorm = np.linalg.norm(g)
        fx = f(x[0], x[1], gamma)

        print(f"k={k:02d}  x=({x[0]:.4f},{x[1]:.4f})  f={fx:.6f}  ||g||={gnorm:.3e}")

        if gnorm < eps:
            break

        x = x - alpha*g
        path.append(x.copy())

    return np.array(path)

#plot de los contornos y los pasos
def plot_contornos_y_pasos(path, gamma):
    x1 = np.linspace(-12, 12, 400)
    x2 = np.linspace(-6, 6, 300)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2, gamma)

    plt.figure(figsize=(9,4.8))
    plt.contour(X1, X2, Z, levels=15)
    plt.plot(path[0,0], path[0,1], 'ro', markersize=8, label="Inicio")
    plt.plot(path[:,0], path[:,1], marker="o", linewidth=2)
    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(f"Gradient Descent | gamma={gamma}")
    plt.show()

def random_x0(x1_range=(-12,12), x2_range=(-6,6)):
    x1 = np.random.uniform(*x1_range)
    x2 = np.random.uniform(*x2_range)
    return (x1, x2)
#Prueba
gamma = 20.0         #forma del valle
alpha = 0.08         #paso que se ajusta
#coordenada inicial la generamos aleatoriamente
x0 = random_x0()
print("Punto inicial aleatorio:", x0)
path = gd_path(x0, alpha, gamma, max_iter=60, eps=1e-6)
print("Pasos ejecutados:", len(path)-1)
print("Ultimo punto:", path[-1])
#imprime el camino recorrido
plot_contornos_y_pasos(path, gamma)
