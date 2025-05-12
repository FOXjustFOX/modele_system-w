import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D # Dla wykresów 3D

# Ustawienie stylu wykresów dla lepszej czytelności (opcjonalnie)
plt.style.use('ggplot')

# --- Definicje Układów ---

def lotka_volterra_rhs(t, state, a, b, c, d):
    """
    Definiuje układ równań Lotki-Volterry.
    state: [x, y] - populacja ofiar i drapieżników
    t: czas (nieużywany jawnie w tym układzie, ale wymagany przez solve_ivp)
    a, b, c, d: parametry układu
    """
    x, y = state
    dxdt = (a - b * y) * x
    dydt = (c * x - d) * y
    return [dxdt, dydt]

def lorenz_rhs(t, state, sigma, rho, beta):
    """
    Definiuje układ równań Lorenza.
    state: [x, y, z]
    t: czas (nieużywany jawnie w tym układzie, ale wymagany przez solve_ivp)
    sigma, rho, beta: parametry układu
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# --- Implementacja Metody Eulera ---

def euler_solver(rhs_func, y0, t_span, dt, args_rhs=()):
    """
    Implementacja metody Eulera do rozwiązywania układów równań różniczkowych.
    rhs_func: funkcja definiująca prawą stronę układu (np. lotka_volterra_rhs)
    y0: warunki początkowe [lista lub array]
    t_span: krotka (t_start, t_end) definiująca przedział czasu
    dt: krok czasowy
    args_rhs: krotka dodatkowych argumentów dla rhs_func (parametry układu)
    """
    t_start, t_end = t_span
    num_steps = int((t_end - t_start) / dt)
    
    t_values = np.linspace(t_start, t_end, num_steps + 1)
    y_values = np.zeros((len(y0), num_steps + 1))
    y_values[:, 0] = y0
    
    for i in range(num_steps):
        current_y = y_values[:, i]
        # Dla funkcji rhs_func, która akceptuje (t, y, *args)
        derivatives = np.array(rhs_func(t_values[i], current_y, *args_rhs))
        y_values[:, i+1] = current_y + dt * derivatives
        
    return t_values, y_values

# Dodajemy funkcję pomocniczą do bezpiecznego ustawiania limitów osi
def safe_set_lim(axis, is_x_axis, data, default_min=-50, default_max=50):
    """
    Bezpiecznie ustawia limity osi, obsługując przypadki NaN i Inf
    """
    # Filtrujemy wartości, żeby usunąć NaN i Inf
    valid_data = data[np.isfinite(data)]
    
    if len(valid_data) == 0:  # Jeśli wszystkie wartości są NaN/Inf
        min_val, max_val = default_min, default_max
    else:
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)
        
        # Ustalamy rozsądne granice
        min_val = default_min if data_min < -1e3 or np.isnan(data_min) else data_min
        max_val = default_max if data_max > 1e3 or np.isnan(data_max) else data_max
        
        # Dodatkowa kontrola wartości
        if not np.isfinite(min_val): min_val = default_min
        if not np.isfinite(max_val): max_val = default_max
        
        # Jeśli min > max (co może się zdarzyć po filtracji), zamieniamy je
        if min_val > max_val:
            min_val, max_val = default_min, default_max
    
    # Ustawiamy limity osi
    if is_x_axis:
        axis.set_xlim(left=min_val, right=max_val)
    else:
        axis.set_ylim(bottom=min_val, top=max_val)

# --- Parametry i Warunki Początkowe ---

# Lotka-Volterra
a, b, c, d = 1.2, 0.6, 0.3, 0.8
x0_lv, y0_lv = 2, 1
initial_state_lv = [x0_lv, y0_lv]
t_span_lv = (0, 25)
dt_values_lv = [0.3, 0.1, 0.01] # Zgodnie z Rysunkiem 1

# Lorenz
sigma, rho, beta = 10, 28, 8/3
x0_l, y0_l, z0_l = 1, 1, 1
initial_state_lorenz = [x0_l, y0_l, z0_l]
t_span_lorenz = (0, 25) # W przykładach t_end jest 25, ale w opisie fig 2 jest t_end=25, a w Fig 4,5 t_end=25. Użyjmy 25.
dt_values_lorenz = [0.03, 0.02, 0.01] # Zgodnie z Rysunkiem 2

# --- Symulacja i Wykresy: Układ Lotki-Volterry ---

print("--- Układ Lotki-Volterry ---")

# 1. Metoda Eulera dla Lotki-Volterry
print("\n1. Metoda Eulera:")
fig_lv_euler, axes_lv_euler = plt.subplots(len(dt_values_lv), 1, figsize=(10, 5 * len(dt_values_lv)), sharex=True)
if len(dt_values_lv) == 1: # Jeśli tylko jeden dt, axes nie jest listą
    axes_lv_euler = [axes_lv_euler]

print("Obliczanie średniego błędu aproksymacji dla metody Eulera (Lotka-Volterra):")
# Rozwiązanie referencyjne z scipy (do obliczenia błędu)
# Używamy małego kroku dla t_eval aby mieć gęste rozwiązanie referencyjne
t_eval_ref_lv = np.linspace(t_span_lv[0], t_span_lv[1], 2000) # Gęsta siatka do interpolacji
sol_lv_scipy_ref = solve_ivp(
    lotka_volterra_rhs, t_span_lv, initial_state_lv,
    args=(a, b, c, d), dense_output=True, t_eval=t_eval_ref_lv
)

for i, dt_lv in enumerate(dt_values_lv):
    t_euler_lv, y_euler_lv = euler_solver(lotka_volterra_rhs, initial_state_lv, t_span_lv, dt_lv, args_rhs=(a,b,c,d))
    x_euler_lv, y_euler_lv_pop = y_euler_lv[0,:], y_euler_lv[1,:]
    
    ax = axes_lv_euler[i]
    ax.plot(t_euler_lv, x_euler_lv, label='Ofiary (x) - Euler', color='red')
    ax.plot(t_euler_lv, y_euler_lv_pop, label='Drapieżniki (y) - Euler', color='blue')
    ax.set_title(f'Układ Lotki-Volterry (Metoda Eulera), dt = {dt_lv}')
    ax.set_xlabel('Czas t')
    ax.set_ylabel('Populacja')
    ax.legend()
    ax.grid(True)

    # Obliczanie błędu
    # Interpolacja rozwiązania Scipy do punktów czasowych Eulera
    y_scipy_at_euler_t = sol_lv_scipy_ref.sol(t_euler_lv)
    error_x = np.mean(np.abs(x_euler_lv - y_scipy_at_euler_t[0,:]))
    error_y = np.mean(np.abs(y_euler_lv_pop - y_scipy_at_euler_t[1,:]))
    avg_error = (error_x + error_y) / 2
    print(f"  dt = {dt_lv}: Średni błąd (x): {error_x:.4e}, Średni błąd (y): {error_y:.4e}, Łączny średni: {avg_error:.4e}")
    
    # Sprawdzenie czy rozwiązanie jest rozbieżne (prosty test)
    if np.any(np.abs(x_euler_lv) > 1e6) or np.any(np.abs(y_euler_lv_pop) > 1e6):
        print(f"    UWAGA: Rozwiązanie dla dt={dt_lv} może być rozbieżne.")
        ax.set_ylim(bottom=0, top=min(np.max(y_euler_lv_pop)*1.1 if np.max(y_euler_lv_pop) < 1e6 else 50, 50)) # Ograniczenie osi Y dla rozbieżnych

fig_lv_euler.suptitle('Układ Lotki-Volterry zamodelowany przy pomocy metody Eulera', fontsize=16, y=1.02)
fig_lv_euler.tight_layout(rect=[0, 0, 1, 0.98])

# 2. Metoda scipy.integrate dla Lotki-Volterry
print("\n2. Metoda scipy.integrate (solve_ivp):")
dt_scipy_plot_lv = 0.002 # Krok dla t_eval, zgodnie z opisem Fig 3
t_eval_lv_scipy = np.arange(t_span_lv[0], t_span_lv[1] + dt_scipy_plot_lv, dt_scipy_plot_lv)

sol_lv_scipy = solve_ivp(
    lotka_volterra_rhs, t_span_lv, initial_state_lv,
    args=(a, b, c, d), dense_output=True, t_eval=t_eval_lv_scipy
)

fig_lv_scipy, ax_lv_scipy = plt.subplots(1, 1, figsize=(12, 6))
ax_lv_scipy.plot(sol_lv_scipy.t, sol_lv_scipy.y[0,:], label='Ofiary (x) - Scipy', color='red')
ax_lv_scipy.plot(sol_lv_scipy.t, sol_lv_scipy.y[1,:], label='Drapieżniki (y) - Scipy', color='blue')
ax_lv_scipy.set_title(f'Układ Lotki-Volterry (scipy.integrate.solve_ivp), dt_eval = {dt_scipy_plot_lv}')
ax_lv_scipy.set_xlabel('Czas t')
ax_lv_scipy.set_ylabel('Populacja')
ax_lv_scipy.legend()
ax_lv_scipy.grid(True)
fig_lv_scipy.tight_layout()


# --- Symulacja i Wykresy: Układ Lorenza ---
print("\n\n--- Układ Lorenza ---")

# 1. Metoda Eulera dla Lorenza
print("\n1. Metoda Eulera:")
num_dt_lorenz = len(dt_values_lorenz)
# Wykresy 2D (rzuty) - jak na Rysunku 2
fig_lorenz_euler_2d, axes_lorenz_euler_2d = plt.subplots(num_dt_lorenz, 3, figsize=(15, 5 * num_dt_lorenz))
if num_dt_lorenz == 1: # Poprawka dla pojedynczego dt
    axes_lorenz_euler_2d = np.array([axes_lorenz_euler_2d])


# Wykresy 3D - jeden dla każdego dt Eulera
figs_lorenz_euler_3d = []
axes_lorenz_euler_3d = []
for _ in range(num_dt_lorenz):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    figs_lorenz_euler_3d.append(fig)
    axes_lorenz_euler_3d.append(ax)


print("Obliczanie średniego błędu aproksymacji dla metody Eulera (Lorenz):")
# Rozwiązanie referencyjne z scipy (do obliczenia błędu)
t_eval_ref_lorenz = np.linspace(t_span_lorenz[0], t_span_lorenz[1], 5000) # Gęstsza siatka
sol_lorenz_scipy_ref = solve_ivp(
    lorenz_rhs, t_span_lorenz, initial_state_lorenz,
    args=(sigma, rho, beta), dense_output=True, t_eval=t_eval_ref_lorenz
)

for i, dt_lorenz in enumerate(dt_values_lorenz):
    t_euler_lorenz, y_euler_lorenz = euler_solver(lorenz_rhs, initial_state_lorenz, t_span_lorenz, dt_lorenz, args_rhs=(sigma, rho, beta))
    x_e, y_e, z_e = y_euler_lorenz[0,:], y_euler_lorenz[1,:], y_euler_lorenz[2,:]

    # Wykresy 2D
    # y(x)
    axes_lorenz_euler_2d[i, 0].plot(x_e, y_e, color='red')
    axes_lorenz_euler_2d[i, 0].set_xlabel('x')
    axes_lorenz_euler_2d[i, 0].set_ylabel('y')
    axes_lorenz_euler_2d[i, 0].set_title(f'y(x), dt = {dt_lorenz}')
    axes_lorenz_euler_2d[i, 0].grid(True)
    # z(x)
    axes_lorenz_euler_2d[i, 1].plot(x_e, z_e, color='red')
    axes_lorenz_euler_2d[i, 1].set_xlabel('x')
    axes_lorenz_euler_2d[i, 1].set_ylabel('z')
    axes_lorenz_euler_2d[i, 1].set_title(f'z(x), dt = {dt_lorenz}')
    axes_lorenz_euler_2d[i, 1].grid(True)
    # z(y)
    axes_lorenz_euler_2d[i, 2].plot(y_e, z_e, color='red')
    axes_lorenz_euler_2d[i, 2].set_xlabel('y')
    axes_lorenz_euler_2d[i, 2].set_ylabel('z')
    axes_lorenz_euler_2d[i, 2].set_title(f'z(y), dt = {dt_lorenz}')
    axes_lorenz_euler_2d[i, 2].grid(True)

    # Sprawdzenie czy rozwiązanie jest rozbieżne
    if np.any(np.abs(x_e) > 1e3) or np.any(np.abs(y_e) > 1e3) or np.any(np.abs(z_e) > 1e3) or \
       np.any(np.isnan(x_e)) or np.any(np.isnan(y_e)) or np.any(np.isnan(z_e)):
         print(f"    UWAGA: Rozwiązanie Lorenza dla dt={dt_lorenz} może być rozbieżne lub zawierać wartości NaN.")
         # Bezpieczne ustawienie limitów osi dla lepszej wizualizacji
         for k_ax in range(3):
            if k_ax == 0:  # wykres y(x)
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], True, x_e)  # x-axis
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], False, y_e) # y-axis
            elif k_ax == 1:  # wykres z(x)
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], True, x_e)  # x-axis
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], False, z_e) # y-axis
            else:  # wykres z(y)
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], True, y_e)  # x-axis
                safe_set_lim(axes_lorenz_euler_2d[i, k_ax], False, z_e) # y-axis

    # Wykres 3D
    ax_3d = axes_lorenz_euler_3d[i]
    ax_3d.plot(x_e, y_e, z_e, lw=0.7, color='red')
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title(f'Układ Lorenza 3D (Euler), dt = {dt_lorenz}')
    figs_lorenz_euler_3d[i].suptitle(f'Układ Lorenza 3D (Euler), dt = {dt_lorenz}', fontsize=16)
    figs_lorenz_euler_3d[i].tight_layout(rect=[0,0,1,0.96])

    # Obliczanie błędu
    y_scipy_at_euler_t_lorenz = sol_lorenz_scipy_ref.sol(t_euler_lorenz)
    error_x_l = np.mean(np.abs(x_e - y_scipy_at_euler_t_lorenz[0,:]))
    error_y_l = np.mean(np.abs(y_e - y_scipy_at_euler_t_lorenz[1,:]))
    error_z_l = np.mean(np.abs(z_e - y_scipy_at_euler_t_lorenz[2,:]))
    avg_error_l = (error_x_l + error_y_l + error_z_l) / 3
    print(f"  dt = {dt_lorenz}: Śr. błąd (x): {error_x_l:.4e}, (y): {error_y_l:.4e}, (z): {error_z_l:.4e}, Łączny średni: {avg_error_l:.4e}")


fig_lorenz_euler_2d.suptitle('Układ Lorenza zamodelowany przy pomocy metody Eulera (rzuty 2D)', fontsize=16, y=1.02)
fig_lorenz_euler_2d.tight_layout(rect=[0, 0, 1, 0.98])


# 2. Metoda scipy.integrate dla Lorenza
print("\n2. Metoda scipy.integrate (solve_ivp):")
dt_scipy_plot_lorenz = 0.002 # Krok dla t_eval, zgodnie z opisem Fig 4, 5
t_eval_lorenz_scipy = np.arange(t_span_lorenz[0], t_span_lorenz[1] + dt_scipy_plot_lorenz, dt_scipy_plot_lorenz)

sol_lorenz_scipy = solve_ivp(
    lorenz_rhs, t_span_lorenz, initial_state_lorenz,
    args=(sigma, rho, beta), dense_output=True, t_eval=t_eval_lorenz_scipy
)
x_s, y_s, z_s = sol_lorenz_scipy.y[0,:], sol_lorenz_scipy.y[1,:], sol_lorenz_scipy.y[2,:]

# Wykresy 2D (rzuty) - jak na Rysunku 4
fig_lorenz_scipy_2d, axes_lorenz_scipy_2d = plt.subplots(3, 1, figsize=(8, 15))
# y(x)
axes_lorenz_scipy_2d[0].plot(x_s, y_s, color='red', lw=0.7)
axes_lorenz_scipy_2d[0].set_xlabel('x')
axes_lorenz_scipy_2d[0].set_ylabel('y')
axes_lorenz_scipy_2d[0].set_title('y(x) - Scipy')
axes_lorenz_scipy_2d[0].grid(True)
# z(x)
axes_lorenz_scipy_2d[1].plot(x_s, z_s, color='red', lw=0.7)
axes_lorenz_scipy_2d[1].set_xlabel('x')
axes_lorenz_scipy_2d[1].set_ylabel('z')
axes_lorenz_scipy_2d[1].set_title('z(x) - Scipy')
axes_lorenz_scipy_2d[1].grid(True)
# z(y)
axes_lorenz_scipy_2d[2].plot(y_s, z_s, color='red', lw=0.7)
axes_lorenz_scipy_2d[2].set_xlabel('y')
axes_lorenz_scipy_2d[2].set_ylabel('z')
axes_lorenz_scipy_2d[2].set_title('z(y) - Scipy')
axes_lorenz_scipy_2d[2].grid(True)

fig_lorenz_scipy_2d.suptitle('Układ Lorenza (scipy.integrate) - rzuty 2D', fontsize=16, y=1.01)
fig_lorenz_scipy_2d.tight_layout(rect=[0, 0, 1, 0.99])

# Wykres 3D - jak na Rysunku 5
fig_lorenz_scipy_3d = plt.figure(figsize=(10, 8))
ax_lorenz_scipy_3d = fig_lorenz_scipy_3d.add_subplot(111, projection='3d')
ax_lorenz_scipy_3d.plot(x_s, y_s, z_s, lw=0.7, color='red')
ax_lorenz_scipy_3d.set_xlabel("X")
ax_lorenz_scipy_3d.set_ylabel("Y")
ax_lorenz_scipy_3d.set_zlabel("Z")
ax_lorenz_scipy_3d.set_title(f'Wizualizacja z(x,y) układu Lorenza (Scipy), dt_eval = {dt_scipy_plot_lorenz}')
fig_lorenz_scipy_3d.tight_layout()

plt.show()

print("\nZakończono symulacje.")