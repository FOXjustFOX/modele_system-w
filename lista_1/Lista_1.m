clc; clear; close all; % zamykamy wszystko

% Warunki początkowe
x_0 = 2;   % Początkowa populacja ofiar
y_0 = 1;   % Początkowa populacja drapieżników

% Parametry symulacji
T = 20;          % Czas symulacji
dt = 0.0001;      % Krok czasowy
N = T/dt;        % Liczba kroków

% Inicjalizacja wektorów
t = linspace(0, T, N); % Czas

% Definicja różnych zestawów parametrów do testowania
parameter_sets = {
    [1.2, 0.6, 0.3, 0.8], % Bazowe
    [2.0, 0.6, 0.3, 0.8], % Zmienione a
    [1.2, 0.6, 0.3, 0.6], % Zmienione d
    [1.2, 1.0, 0.3, 0.8], % Zmienione b
    [1.2, 0.6, 0.5, 0.8]  % Zmienione c
};

results = zeros(length(parameter_sets), 3); % [max_ofiar, max_dra, okres]

for p = 1:length(parameter_sets)
    a = parameter_sets{p}(1);
    b = parameter_sets{p}(2);
    c = parameter_sets{p}(3);
    d = parameter_sets{p}(4);
    
    % Inicjalizacja wektorów
    x = zeros(1, N);       % Populacja ofiar
    y = zeros(1, N);       % Populacja drapieżników

    % Ustawienie wartości początkowych
    x(1) = x_0;
    y(1) = y_0;

    % Metoda Eulera
    for i = 1:N-1
        dx = (a - b*y(i)) * x(i);
        dy = (c * x(i) - d) * y(i);
        
        x(i+1) = x(i) + dx * dt;
        y(i+1) = y(i) + dy * dt;
    end

    % Na końcu zapisujemy wyniki
    results(p, 1) = max(x);
    results(p, 2) = max(y);

    % Znajdowanie lokalnych maksimów (punktów gdzie zmienia się znak pochodnej z dodatniej na ujemną)
    dx_dt = diff(x);
    sign_changes = diff(sign(dx_dt));
    peak_indices = find(sign_changes < 0) + 1; % +1 aby uzyskać indeks rzeczywistego maksimum

    % Obliczenie średniego okresu
    if length(peak_indices) >= 2
        periods = diff(t(peak_indices));
        avg_period = mean(periods);
    else
        avg_period = NaN; % Jeśli nie znaleziono co najmniej 2 szczytów
    end

    results(p, 3) = avg_period; % Okres oscylacji
    
    figure;
    plot(t, x, 'b', 'LineWidth', 1.5); hold on;
    plot(t, y, 'r', 'LineWidth', 1.5);
    xlabel('Czas');
    ylabel('Populacja');
    legend('Ofiary', 'Drapieżnicy');
    title('Model Lotki-Volterry');
    grid on;

    saveas(gcf, 'rys_'+string(p)+'.png');

end

fprintf('Wyniki: [max_ofiary, max_drapieżnicy, okres]\n');
disp(results);

% Wykres populacji w czasie dla ostatniego zestawu parametrów
