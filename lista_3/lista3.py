import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  # Do tworzenia wykresów 3D


# ==== Definicje układów różniczkowych ====
# Układ Lotki-Volterry: 
def lotka_volterra(X, t, a=1.2, b=0.6, c=0.3, d=0.8):
    x, y = X
    dxdt = (a - b * y) * x
    dydt = (c * x - d) * y
    return [dxdt, dydt]


# Układ Lorenza:
def lorenz(X, t, sigma=10, beta=8 / 3, rho=28):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


# ==== Własna implementacja metody Eulera ====
def euler(f, X0, t):
    if len(t) < 2:
         return np.array([X0]) if len(t) > 0 else np.array([])

    dt = t[1] - t[0]  # krok czasowy (zakładamy stały krok)
    X = np.zeros((len(t), len(X0)))  # macierz na wynik
    X[0] = np.array(X0)  # warunki początkowe

    for i in range(1, len(t)):
        dX = np.array(f(X[i - 1], t[i - 1]))

        if np.any(np.isnan(dX)) or np.any(np.isinf(dX)) or np.any(np.abs(dX) > 1e10):
             print(f"Euler warning/error in step {i} for dt={dt}: dX = {dX}. Truncating result.")
             return X[:i]

        X[i] = X[i - 1] + dt * dX

        if (
            np.any(np.isnan(X[i]))
            or np.any(np.isinf(X[i]))
            or np.any(np.abs(X[i]) > 1e10)
        ):
            print(f"Euler warning/error in step {i} for dt={dt}: X = {X[i]}. Truncating result.")
            return X[:i]

    return X


# ==== Funkcja do obliczania błędu aproksymacji dla metody Eulera ====
def calculate_euler_error(system, initial_conditions, t_values):
    euler_solution = euler(system, initial_conditions, t_values)

    if len(euler_solution) == 0:
         print("Euler simulation returned no points. Cannot calculate error.")
         return np.nan, np.array([]), np.array([])

    t_truncated = t_values[: len(euler_solution)]

    odeint_solution = odeint(system, initial_conditions, t_truncated, rtol=1e-8, atol=1e-8)

    #odl euklidesowa
    errors = np.sqrt(
        np.sum((euler_solution - odeint_solution) ** 2, axis=1)
    )

    mean_error = np.mean(errors) if len(errors) > 0 else np.nan

    return mean_error, errors, t_truncated


# ==== LOTKA-VOLTERRA: Euler – wykresy dla różnych kroków dt ====
dt_values_lv = [0.001, 0.03, 0.015, 0.3]
t_max_lv = 25
initial_conditions_lv = [2, 1]

print("==== Lotka-Volterra: Obliczanie i wyświetlanie błędów aproksymacji ====")
lv_errors = {}
fig_lv_error = plt.figure(figsize=(10, 6))
plt.title("Błędy aproksymacji metody Eulera dla układu Lotki-Volterry")

dt_values_lv_sorted = sorted(dt_values_lv)

for dt in dt_values_lv_sorted:
    t = np.arange(0, t_max_lv + dt/2, dt)
    mean_error, errors, t_truncated = calculate_euler_error(
        lotka_volterra, initial_conditions_lv, t
    )
    lv_errors[dt] = mean_error

    if not np.isnan(mean_error):
        print(f"dt = {dt}: Średni błąd = {mean_error:.6e}")
        plt.plot(t_truncated, errors, label=f"dt = {dt}")
    else:
         print(f"dt = {dt}: Nie można obliczyć błędu (symulacja Eulera nie powiodła się lub brak danych)")

plt.xlabel("Czas [t]")
plt.ylabel("Błąd (odległość euklidesowa) [skala log]")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("\n==== Lotka-Volterra: Wykresy populacji (Euler) ====")
for dt in dt_values_lv_sorted:
    t = np.arange(0, t_max_lv + dt/2, dt)
    sol = euler(lotka_volterra, initial_conditions_lv, t)

    if sol.shape[0] < 2:
        print(f"Symulacja Lotka-Volterra Euler dla dt={dt} nie dała wystarczających wyników do wykreślenia.")
        continue

    plt.figure()
    plt.title(f"Lotka-Volterra – Euler (dt = {dt})")
    plt.plot(t[:sol.shape[0]], sol[:, 0], label="Ofiary (x)", color="blue")
    plt.plot(t[:sol.shape[0]], sol[:, 1], label="Drapieżniki (y)", color="red")
    plt.xlabel("Czas [t]")
    plt.ylabel("Populacja")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

print("\n==== Lotka-Volterra: Wykres populacji (odeint) ====")
t_odeint_lv = np.linspace(0, t_max_lv, 500)
sol_odeint_lv = odeint(lotka_volterra, initial_conditions_lv, t_odeint_lv)

plt.figure()
plt.title("Lotka-Volterra – odeint")
plt.plot(t_odeint_lv, sol_odeint_lv[:, 0], label="Ofiary (x)", color="blue")
plt.plot(t_odeint_lv, sol_odeint_lv[:, 1], label="Drapieżniki (y)", color="red")
plt.xlabel("Czas [t]")
plt.ylabel("Populacja")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==== LORENZ: Obliczanie błędów dla różnych dt ====
dt_values_lorenz = [0.3, 0.005, 0.001]
t_max_lorenz = 25
initial_conditions_lorenz = [1, 1, 1]

print("\n==== Lorenz: Obliczanie i wyświetlanie średnich błędów aproksymacji ====")
lorenz_errors = {}

for dt in dt_values_lorenz:
    t = np.arange(0, t_max_lorenz + dt/2, dt)
    mean_error, errors, t_truncated = calculate_euler_error(
        lorenz, initial_conditions_lorenz, t
    )
    lorenz_errors[dt] = mean_error
    if not np.isnan(mean_error):
         print(f"dt = {dt}: Średni błąd = {mean_error:.6e}")
    else:
         print(f"dt = {dt}: Nie można obliczyć średniego błędu (symulacja Eulera nie powiodła się lub brak danych)")


# ==== LORENZ: Wykres błędu metody Eulera w czasie ====
print("\n==== Lorenz: Generowanie wykresu błędów w czasie (Euler vs odeint) ====")
fig_lorenz_error = plt.figure(figsize=(10, 6))
plt.title("Błędy aproksymacji metody Eulera dla układu Lorenza")

dt_values_lorenz_sorted = sorted(dt_values_lorenz, reverse=True)

for dt in dt_values_lorenz_sorted:
    t = np.arange(0, t_max_lorenz + dt/2, dt)
    mean_error, errors, t_truncated = calculate_euler_error(
        lorenz, initial_conditions_lorenz, t
    )

    if not np.isnan(mean_error):
        plt.plot(t_truncated, errors, label=f"dt = {dt} (Średni błąd: {mean_error:.2e})")
    else:
        print(f"Nie można wykreślić błędu dla dt={dt}, symulacja Eulera nie powiodła się lub brak danych.")

plt.xlabel("Czas [t]")
plt.ylabel("Błąd (odległość euklidesowa) [skala log]")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ==== LORENZ: Euler – rzuty 2D dla różnych dt (każdy zestaw w osobnym oknie) ====
print("\n==== Lorenz: Generowanie wykresów rzutów 2D (Euler) dla różnych dt ====")

for dt in dt_values_lorenz_sorted:
    t = np.arange(0, t_max_lorenz + dt/2, dt)
    sol = euler(lorenz, initial_conditions_lorenz, t)

    if sol.shape[0] < 2:
         print(f"Symulacja Lorenz Euler dla dt={dt} nie dała wystarczających wyników do wykreślenia rzutów 2D.")
         continue

    x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

    # Tworzymy nową figurę dla tego konkretnego dt
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"Układ Lorenza – Euler (rzuty 2D, dt = {dt})", fontsize=16)

    # Rzut x vs y
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(x, y, color="firebrick", linewidth=1)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("x vs y")
    ax1.grid(True)

    # Rzut x vs z
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(x, z, color="firebrick", linewidth=1)
    ax2.set_xlabel("x")
    ax2.set_ylabel("z")
    ax2.set_title("x vs z")
    ax2.grid(True)

    # Rzut y vs z
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(y, z, color="firebrick", linewidth=1)
    ax3.set_xlabel("y")
    ax3.set_ylabel("z")
    ax3.set_title("y vs z")
    ax3.grid(True)

    # Dopasowanie układu wykresów i wyświetlenie figury
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# ==== LORENZ: odeint – rzuty 2D ====
print("\n==== Lorenz: Wykresy rzutów 2D (odeint) ====")
t_odeint_lorenz = np.linspace(0, t_max_lorenz, 5000)
sol_odeint_lorenz = odeint(lorenz, initial_conditions_lorenz, t_odeint_lorenz)
x_odeint, y_odeint, z_odeint = (
    sol_odeint_lorenz[:, 0],
    sol_odeint_lorenz[:, 1],
    sol_odeint_lorenz[:, 2],
)

fig_odeint_2d, axes_odeint_2d = plt.subplots(1, 3, figsize=(15, 5))
fig_odeint_2d.suptitle(
    "Układ Lorenza zamodelowany przy pomocy odeint", fontsize=16
)

axes_odeint_2d[0].plot(x_odeint, y_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[0].set_xlabel("x")
axes_odeint_2d[0].set_ylabel("y")
axes_odeint_2d[0].set_title("x vs y")
axes_odeint_2d[0].grid(True)

axes_odeint_2d[1].plot(x_odeint, z_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[1].set_xlabel("x")
axes_odeint_2d[1].set_ylabel("z")
axes_odeint_2d[1].set_title("x vs z")
axes_odeint_2d[1].grid(True)

axes_odeint_2d[2].plot(y_odeint, z_odeint, color="mediumvioletred", linewidth=1)
axes_odeint_2d[2].set_xlabel("y")
axes_odeint_2d[2].set_ylabel("z")
axes_odeint_2d[2].set_title("y vs z")
axes_odeint_2d[2].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==== LORENZ: Euler – wykresy 3D dla różnych dt ====
print("\n==== Lorenz: Generowanie wykresów 3D (Euler) dla różnych dt ====")

for dt in dt_values_lorenz_sorted:
    t = np.arange(0, t_max_lorenz + dt/2, dt)
    sol_euler = euler(lorenz, initial_conditions_lorenz, t)

    if sol_euler.shape[0] < 2:
        print(f"Symulacja Lorenz Euler 3D dla dt={dt} nie dała wystarczających wyników do wykreślenia.")
        continue

    x_euler, y_euler, z_euler = sol_euler[:, 0], sol_euler[:, 1], sol_euler[:, 2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x_euler, y_euler, z_euler, color="darkblue", linewidth=0.8)
    ax.set_title(f"Układ Lorenza – metoda Eulera (3D, dt={dt})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()

# ==== LORENZ: odeint – wykres 3D ====
print("\n==== Lorenz: Wykres 3D (odeint) ====")
fig_odeint_3d = plt.figure(figsize=(10, 8))
ax_odeint_3d = fig_odeint_3d.add_subplot(111, projection="3d")
ax_odeint_3d.plot(x_odeint, y_odeint, z_odeint, color="indigo", linewidth=0.8)
ax_odeint_3d.set_title("Układ Lorenza – odeint (3D)")
ax_odeint_3d.set_xlabel("x")
ax_odeint_3d.set_ylabel("y")
ax_odeint_3d.set_zlabel("z")
plt.tight_layout()
plt.show()

print("\n==== Symulacje zakończone ====")