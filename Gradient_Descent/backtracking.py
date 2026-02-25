import numpy as np
import matplotlib.pyplot as plt

def f(x):
    x1, x2 = x
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)

def grad_f(x):
    x1, x2 = x
    e1 = np.exp(x1 + 3*x2 - 0.1)
    e2 = np.exp(x1 - 3*x2 - 0.1)
    e3 = np.exp(-x1 - 0.1)
    df_dx1 = e1 + e2 - e3
    df_dx2 = 3*e1 - 3*e2
    return np.array([df_dx1, df_dx2], dtype=float)

#Backtracking line search
def backtracking(x, p, c=1e-4, rho=0.5, t0=1.0, max_backtracks=50):
    #Busca t tal que: f(x + t p) <= f(x) + c t grad(x)^T p
    t = t0
    fx = f(x)
    g = grad_f(x)
    gp = np.dot(g, p)  # g^T p (debe ser negativo si p es descenso)

    # Si por alguna razón p no es dirección de descenso, se hace p=-g
    if gp >= 0:
        p = -g
        gp = np.dot(g, p)

    for _ in range(max_backtracks):
        if f(x + t*p) <= fx + c*t*gp:
            return t
        t *= rho

    return t  # si no encontró, regresa el t chiquito que quedó

#Gradient descent con backtracking
def gd_backtracking(x0, max_iter=50, eps=1e-6, c=1e-4, rho=0.5, t0=1.0):
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    steps = []

    for k in range(max_iter):
        g = grad_f(x)
        gnorm = np.linalg.norm(g)
        if gnorm < eps:
            break

        p = -g
        t = backtracking(x, p, c=c, rho=rho, t0=t0)
        x = x + t*p

        path.append(x.copy())
        steps.append(t)

    return np.array(path), np.array(steps)

#Contornos + trayectoria
def plot_contornos_y_pasos(path, margin=0.6):
    x1_min, x1_max = path[:,0].min()-margin, path[:,0].max()+margin
    x2_min, x2_max = path[:,1].min()-margin, path[:,1].max()+margin

    x1 = np.linspace(x1_min, x1_max, 400)
    x2 = np.linspace(x2_min, x2_max, 300)
    X1, X2 = np.meshgrid(x1, x2)

    Z = np.exp(X1 + 3*X2 - 0.1) + np.exp(X1 - 3*X2 - 0.1) + np.exp(-X1 - 0.1)

    plt.figure(figsize=(9,4.8))
    plt.contour(X1, X2, np.log(Z), levels=20)
    plt.plot(path[:,0], path[:,1], marker="o", linewidth=2)
    plt.scatter(path[0,0], path[0,1], s=70, marker="s", label="inicio")
    plt.scatter(path[-1,0], path[-1,1], s=90, marker="X", label="final")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Backtracking line search")
    plt.legend()
    plt.show()

#Prueba
if __name__ == "__main__":
    #np.random.seed(42)
    #x0 = np.random.uniform(-5, 5, size=2)
    x0=(1.8,0.8)
    path, steps = gd_backtracking(x0, max_iter=60, eps=1e-8, c=1e-4, rho=0.5, t0=1.0)

    print("Iteraciones:", len(path) - 1)
    print("Último punto:", path[-1])
    print("Último f(x):", f(path[-1]))
    if len(steps) > 0:
        print("Paso mínimo usado:", steps.min(), " | paso máximo usado:", steps.max())

    plot_contornos_y_pasos(path)