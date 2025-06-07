import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
import os
import pandas as pd
import time # Dodajemy do pomiaru czasu

# --- Krok 1: Utworzenie folderu na wyniki ---
output_dir = "lista_4/wyniki"

# --- Krok 2: Funkcja do symulacji i rysowania ---
def run_and_plot_simulation(params, t_span=(0, 25), num_points=1000):
    """
    Uruchamia symulację dla danych parametrów, rysuje wykresy i zapisuje je do plików.
    """
    # Rozpakowanie parametrów
    r, K, N0, label = params['r'], params['K'], params['N0'], params['label']
    
    filename_safe_label = label.lower().replace(" ", "_").replace(":", "")

    # Definicja równania dla SciPy
    def logistic_growth_rhs(t, N, r_val, K_val):
        return r_val * N * (1 - N / K_val)

    t_eval = np.linspace(*t_span, num_points)

    # --- Rozwiązanie analityczne (SymPy) ---
    t_sym, r_sym, K_sym, N0_sym = sp.symbols('t r K N0')
    N_sym = sp.Function('N')
    ode = sp.Eq(N_sym(t_sym).diff(t_sym), r_sym * N_sym(t_sym) * (1 - N_sym(t_sym) / K_sym))
    exact_solution_expr = sp.dsolve(ode, ics={N_sym(0): N0_sym}).rhs
    
    # Podstawienie wartości parametrów (r, K, N0) do wyrażenia
    expr_with_params = exact_solution_expr.subs({r_sym: r, K_sym: K, N0_sym: N0})
    
    start_time = time.time() # Pomiar czasu
    
    N_exact_list = []
    for t_val in t_eval:
        # Podstawiamy konkretną wartość czasu 't_val' do symbolu 't_sym'
        # i konwertujemy wynik (który jest obiektem SymPy) na typ float.
        value = float(expr_with_params.subs(t_sym, t_val))
        N_exact_list.append(value)
        
    # Konwersja listy wyników z powrotem na tablicę NumPy
    N_exact = np.array(N_exact_list)
    
    end_time = time.time()
    print(f"  Czas obliczeń analitycznych: {end_time - start_time:.4f} s")

    # --- Rozwiązanie numeryczne (SciPy) ---
    sol = solve_ivp(
        fun=logistic_growth_rhs, t_span=t_span, y0=[N0], args=(r, K),
        t_eval=t_eval, rtol=1e-8
    )
    N_numeric = sol.y[0]

    # --- Obliczenie błędów i wizualizacja ---
    abs_error = np.abs(N_numeric - N_exact)
    mae = np.mean(abs_error)
    mse = np.mean((N_numeric - N_exact)**2)

    # Wykres 1: Porównanie
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t_eval, N_numeric, 'k-', label='Rozwiązanie numeryczne (SciPy)', linewidth=4)
    ax1.plot(t_eval, N_exact, 'r--', label='Rozwiązanie analityczne (SymPy)', linewidth=2)
    ax1.set_title(f"Model wzrostu logistycznego: {label}")
    ax1.set_xlabel('Czas [t]')
    ax1.set_ylabel('Wielkość populacji [N(t)]')
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, f"porownanie_{filename_safe_label}.png"))
    plt.close(fig1)

    # Wykres 2: Błąd
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_eval, abs_error, label='Błąd bezwzględny', color='blue')
    ax2.set_title(f"Błąd bezwzględny dla: {label}")
    ax2.set_xlabel('Czas [t]')
    ax2.set_ylabel('Błąd bezwzględny')
    ax2.grid(True)
    ax2.set_yscale('log')
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, f"blad_{filename_safe_label}.png"))
    plt.close(fig2)

    return {"Opis": label, "MAE": mae, "MSE": mse}

# --- Scenariusze i uruchomienie ---
scenarios = [
    {'label': '1. Scenariusz bazowy', 'r': 0.5, 'K': 1000, 'N0': 50},
    {'label': '2. Szybki wzrost (r=2.0)', 'r': 2.0, 'K': 1000, 'N0': 50},
    {'label': '3. Spadek populacji (N0 > K)', 'r': 0.5, 'K': 1000, 'N0': 1500},
    {'label': '4. Niska pojemnosc srodowiska (K=500)', 'r': 0.5, 'K': 500, 'N0': 50},
    {'label': '5. Start blisko granicy (N0=950)', 'r': 0.5, 'K': 1000, 'N0': 950}
]

results = []
for params in scenarios:
    print(f"Uruchamiam symulację dla: '{params['label']}'...")
    result = run_and_plot_simulation(params)
    results.append(result)

df_results = pd.DataFrame(results)
print("\nPorównanie błędów dla różnych scenariuszy:")
print(df_results.to_string(index=False))