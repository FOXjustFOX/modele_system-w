import sympy
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Krok 1: Definicja symboli i równania dla sympy ---
t_sym = sympy.symbols('t')
# Np. dla rozpadu promieniotwórczego:
# N_sym = sympy.Function('N')(t_sym)
# lambda_val = 0.1 # Przykładowa wartość parametru
# N0_val = 1000   # Przykładowa wartość warunku początkowego
# ode_sym = sympy.Eq(N_sym.diff(t_sym), -lambda_val * N_sym)
# ics_sym = {N_sym.subs(t_sym, 0): N0_val}

# Np. dla obwodu RC (ładowanie kondensatora Vc(t)):
# Vc_sym = sympy.Function('Vc')(t_sym)
# Vs_val = 5.0
# R_val = 1000.0
# C_val = 0.0001
# V0_val = 0.0
# ode_sym = sympy.Eq(Vc_sym.diff(t_sym), (Vs_val - Vc_sym) / (R_val * C_val))
# ics_sym = {Vc_sym.subs(t_sym, 0): V0_val}

# Wstaw tutaj definicję swojego równania i warunków początkowych
# Pamiętaj, aby zdefiniować wartości parametrów (np. lambda_val, N0_val)

# --- Krok 2: Rozwiązanie analityczne (dokładne) za pomocą sympy ---
# exact_sol_sym = sympy.dsolve(ode_sym, func=N_sym, ics=ics_sym) # dla rozpadu
# exact_sol_sym = sympy.dsolve(ode_sym, func=Vc_sym, ics=ics_sym) # dla RC
# print(f"Rozwiązanie analityczne: {exact_sol_sym}")

# Konwersja rozwiązania sympy na funkcję numeryczną (callable)
# exact_sol_func = sympy.lambdify(t_sym, exact_sol_sym.rhs, 'numpy')

# --- Krok 3: Definicja funkcji dla rozwiązania numerycznego (scipy) ---
# Ta funkcja musi być w formie dy/dt = f(t, y)
# Np. dla rozpadu promieniotwórczego:
# def ode_scipy_decay(t, N, lambda_p):
#     return -lambda_p * N
# params_scipy = (lambda_val,) # Tuple parametrów dla solve_ivp

# Np. dla obwodu RC:
# def ode_scipy_rc(t, Vc, Vs_p, R_p, C_p):
#     return (Vs_p - Vc) / (R_p * C_p)
# params_scipy = (Vs_val, R_val, C_val)

# Wstaw tutaj definicję funkcji dla scipy

# --- Krok 4: Przygotowanie do symulacji numerycznej ---
T_end = 10.0  # Całkowity czas symulacji
dt = 0.1    # Krok czasowy dla wizualizacji błędu
t_eval = np.arange(0, T_end + dt, dt) # Punkty czasowe do ewaluacji

# Warunek początkowy dla scipy (musi być tablicą/listą)
# y0_scipy = [N0_val] # dla rozpadu
# y0_scipy = [V0_val] # dla RC

# --- Krok 5: Rozwiązanie numeryczne (przybliżone) za pomocą scipy.solve_ivp ---
# sol_numeric = solve_ivp(ode_scipy_decay, [0, T_end], y0_scipy, args=params_scipy, dense_output=True, t_eval=t_eval)
# sol_numeric = solve_ivp(ode_scipy_rc, [0, T_end], y0_scipy, args=params_scipy, dense_output=True, t_eval=t_eval)

# numerical_solution_values = sol_numeric.y[0]
# t_numeric = sol_numeric.t

# --- Krok 6: Obliczenie wartości rozwiązania dokładnego w punktach czasowych ---
# exact_solution_values = exact_sol_func(t_numeric)

# --- Krok 7: Porównanie rozwiązań - obliczenie błędów ---
# Upewnij się, że t_numeric i t_eval (użyte do exact_solution_values) są takie same
# if not np.array_equal(t_numeric, t_eval):
#    exact_solution_values = exact_sol_func(t_eval) # Przelicz, jeśli trzeba
#    numerical_solution_values = sol_numeric.sol(t_eval)[0] # Użyj interpolacji, jeśli t_eval różni się od t z solve_ivp
# else:
#    exact_solution_values = exact_sol_func(t_eval)
#    numerical_solution_values = sol_numeric.y[0]


# Mean Absolute Error (MAE)
# mae = np.mean(np.abs(exact_solution_values - numerical_solution_values))
# print(f"Średni błąd bezwzględny (MAE): {mae}")

# Mean Squared Error (MSE)
# mse = np.mean((exact_solution_values - numerical_solution_values)**2)
# print(f"Średni błąd kwadratowy (MSE): {mse}")

# Błąd w każdym kroku
# error_stepwise = exact_solution_values - numerical_solution_values

# --- Krok 8: Wizualizacja ---
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.plot(t_eval, exact_solution_values, label='Rozwiązanie dokładne (sympy)', linestyle='-')
# plt.plot(t_eval, numerical_solution_values, label='Rozwiązanie przybliżone (scipy)', linestyle='--')
# plt.xlabel('Czas t')
# plt.ylabel('Wartość funkcji') # Np. N(t) lub Vc(t)
# plt.title('Porównanie rozwiązań')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(t_eval, error_stepwise, label='Błąd (dokładne - przybliżone)', color='red')
# plt.xlabel('Czas t')
# plt.ylabel('Błąd')
# plt.title('Błąd w każdym kroku czasowym')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# PRZYKŁAD DLA ROZPADU PROMIENIOTWÓRCZEGO (odkomentuj i uzupełnij)
# --- Krok 1: Definicja symboli i równania dla sympy ---
t_sym = sympy.symbols('t')
N_sym = sympy.Function('N')(t_sym)
lambda_val = 0.1
N0_val = 1000
ode_sym = sympy.Eq(N_sym.diff(t_sym), -lambda_val * N_sym)
ics_sym = {N_sym.subs(t_sym, 0): N0_val}

# --- Krok 2: Rozwiązanie analityczne (dokładne) za pomocą sympy ---
exact_sol_sym = sympy.dsolve(ode_sym, func=N_sym, ics=ics_sym)
print(f"Rozwiązanie analityczne: {exact_sol_sym}")
exact_sol_func = sympy.lambdify(t_sym, exact_sol_sym.rhs, 'numpy')

# --- Krok 3: Definicja funkcji dla rozwiązania numerycznego (scipy) ---
def ode_scipy_decay(t, N, lambda_p):
    return -lambda_p * N
params_scipy = (lambda_val,)

# --- Krok 4: Przygotowanie do symulacji numerycznej ---
T_end = 50.0
dt = 0.5
t_eval = np.arange(0, T_end + dt, dt)
y0_scipy = [N0_val]

# --- Krok 5: Rozwiązanie numeryczne (przybliżone) za pomocą scipy.solve_ivp ---
sol_numeric = solve_ivp(ode_scipy_decay, [0, T_end], y0_scipy, args=params_scipy, dense_output=True, t_eval=t_eval)
numerical_solution_values = sol_numeric.y[0]
t_numeric = sol_numeric.t # t_numeric będzie takie samo jak t_eval dzięki opcji t_eval

# --- Krok 6: Obliczenie wartości rozwiązania dokładnego w punktach czasowych ---
exact_solution_values = exact_sol_func(t_numeric)

# --- Krok 7: Porównanie rozwiązań - obliczenie błędów ---
mae = np.mean(np.abs(exact_solution_values - numerical_solution_values))
print(f"Średni błąd bezwzględny (MAE): {mae}")
mse = np.mean((exact_solution_values - numerical_solution_values)**2)
print(f"Średni błąd kwadratowy (MSE): {mse}")
error_stepwise = exact_solution_values - numerical_solution_values

# --- Krok 8: Wizualizacja ---
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(t_numeric, exact_solution_values, label='Rozwiązanie dokładne (sympy)', linestyle='-', marker='o', markersize=4)
plt.plot(t_numeric, numerical_solution_values, label='Rozwiązanie przybliżone (scipy)', linestyle='--', marker='x', markersize=4)
plt.xlabel('Czas t [s]')
plt.ylabel('N(t) [liczba jąder]')
plt.title(f'Rozpad promieniotwórczy (N0={N0_val}, λ={lambda_val})')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_numeric, error_stepwise, label='Błąd (dokładne - przybliżone)', color='red', marker='.')
plt.xlabel('Czas t [s]')
plt.ylabel('Błąd absolutny')
plt.title('Błąd w każdym kroku czasowym')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Możesz teraz wybrać inny przykład, np. obwód RC, i zaimplementować go podobnie.
# Pamiętaj o dostosowaniu równań, parametrów, warunków początkowych i etykiet na wykresach.