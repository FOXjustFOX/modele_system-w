import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  # Do tworzenia wykresów 3D


# ==== Definicje układów różniczkowych ====
# Układ Lotki-Volterry: model drapieżnik-ofiara
def lotka_volterra(X, t, a=1.2, b=0.6, c=0.3, d=0.8):
    x, y = X
    dxdt = (a - b * y) * x
    dydt = (c * x - d) * y
    return [dxdt, dydt]


# Układ Lorenza: model chaotyczny w 3 wymiarach
def lorenz(X, t, sigma=10, beta=8 / 3, rho=28):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# ==== Własna implementacja metody Eulera ====
def euler(f, X0, t):
    dt = t[1] - t[0]  # krok czasowy
    X = np.zeros((len(t), len(X0)))  # macierz na wynik
    X[0] = X0  # warunki początkowe

    for i in range(1, len(t)):
        dX = np.array(f(X[i - 1], t[i - 1]))  # pochodna w punkcie
        # Sprawdzenie, czy pochodna jest poprawna
        if np.any(np.isnan(dX)) or np.any(np.isinf(dX)) or np.any(np.abs(dX) > 1e6):
            print(f"Błąd numeryczny w kroku {i}: dX = {dX}")
            X = X[:i]  # ucięcie wyników do tego momentu
            break

        X[i] = X[i - 1] + dt * dX  # krok Eulera

        # Sprawdzenie poprawności nowego punktu
        if (
            np.any(np.isnan(X[i]))
            or np.any(np.isinf(X[i]))
            or np.any(np.abs(X[i]) > 1e6)
        ):
            print(f"Błąd numeryczny w kroku {i}: X = {X[i]}")
            X = X[:i]
            break

    return X


# ==== LOTKA-VOLTERRA: Euler – wykresy dla różnych kroków dt ====
dt_values_lv = [0.3, 0.1, 0.01]  # trzy różne kroki czasowe
t_max = 25  # czas symulacji

for dt in dt_values_lv:
    t = np.arange(0, t_max, dt)
    sol = euler(lotka_volterra, [2, 1], t)

    # Wykresy x(t) i y(t)
    plt.figure()
    plt.title(f"Lotka-Volterra – Euler (dt = {dt})")
    plt.plot(t, sol[:, 0], label="Ofiary (x)", color="blue")
    plt.plot(t, sol[:, 1], label="Drapieżniki (y)", color="red")
    plt.xlabel("Czas [t]")
    plt.ylabel("Populacja")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==== LOTKA-VOLTERRA: odeint ====
t_odeint_lv = np.linspace(0, t_max, 500)
sol_odeint_lv = odeint(lotka_volterra, [2, 1], t_odeint_lv)

# Wykres Lotki-Volterry z odeint
plt.figure()
plt.title("Lotka-Volterra – odeint")
plt.plot(t_odeint_lv, sol_odeint_lv[:, 0], label="Ofiary (x)", color="lime")
plt.plot(t_odeint_lv, sol_odeint_lv[:, 1], label="Drapieżniki (y)", color="forestgreen")
plt.xlabel("Czas [t]")
plt.ylabel("Populacja")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==== LORENZ: Euler – 9 rzutów trajektorii (x/y/z w parach) ====
dt_values_lorenz = [0.01, 0.005, 0.001]  # zmniejszone dt by zapobiec błędom
t_max = 25

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Układ Lorenza zamodelowany przy pomocy metody Eulera", fontsize=16)

for row_idx, dt in enumerate(dt_values_lorenz):
    t = np.arange(0, t_max, dt)
    sol = euler(lorenz, [1, 1, 1], t)
    x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

    # Rzut x vs y
    ax1 = axes[row_idx, 0]
    ax1.plot(x, y, color="firebrick", linewidth=1)
    ax1.set_title(f"dt = {dt}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # Rzut x vs z
    ax2 = axes[row_idx, 1]
    ax2.plot(x, z, color="firebrick", linewidth=1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")

    # Rzut y vs z
    ax3 = axes[row_idx, 2]
    ax3.plot(y, z, color="firebrick", linewidth=1)
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==== LORENZ: odeint – rzuty 2D ====
t_odeint_lorenz = np.linspace(0, t_max, 5000)
sol_odeint_lorenz = odeint(lorenz, [1, 1, 1], t_odeint_lorenz)
x_odeint, y_odeint, z_odeint = (
    sol_odeint_lorenz[:, 0],
    sol_odeint_lorenz[:, 1],
    sol_odeint_lorenz[:, 2],
)

fig_odeint_2d, axes_odeint_2d = plt.subplots(1, 3, figsize=(15, 5))
fig_odeint_2d.suptitle(
    "Układ Lorenza zamodelowany przy pomocy odeint (rzuty 2D)", fontsize=16
)

# Rzut x vs y
axes_odeint_2d[0].plot(x_odeint, y_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[0].set_xlabel("x")
axes_odeint_2d[0].set_ylabel("y")
axes_odeint_2d[0].set_title("x vs y")

# Rzut x vs z
axes_odeint_2d[1].plot(x_odeint, z_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[1].set_xlabel("x")
axes_odeint_2d[1].set_ylabel("z")
axes_odeint_2d[1].set_title("x vs z")

# Rzut y vs z
axes_odeint_2d[2].plot(y_odeint, z_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[2].set_xlabel("y")
axes_odeint_2d[2].set_ylabel("z")
axes_odeint_2d[2].set_title("y vs z")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==== LORENZ: Euler – wykres 3D ====
# Dodatkowa wizualizacja 3D układu Lorenza przy użyciu metody Eulera
dt_euler_3d = 0.01  # stabilny krok czasowy
t = np.arange(0, t_max, dt_euler_3d)
sol_euler = euler(lorenz, [1, 1, 1], t)
x_euler, y_euler, z_euler = sol_euler[:, 0], sol_euler[:, 1], sol_euler[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")  # tworzymy wykres 3D
ax.plot(x_euler, y_euler, z_euler, color="darkblue", linewidth=0.8)
ax.set_title("Układ Lorenza – metoda Eulera (3D)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.tight_layout()
plt.show()

# ==== LORENZ: odeint – wykres 3D ====
fig_odeint_3d = plt.figure(figsize=(10, 8))
ax_odeint_3d = fig_odeint_3d.add_subplot(111, projection="3d")
ax_odeint_3d.plot(x_odeint, y_odeint, z_odeint, color="indigo", linewidth=0.8)
ax_odeint_3d.set_title("Układ Lorenza – odeint (3D)")
ax_odeint_3d.set_xlabel("x")
ax_odeint_3d.set_ylabel("y")
ax_odeint_3d.set_zlabel("z")
plt.tight_layout()
plt.show()
