clc; clear; close all; % zamykamy wszystko

% Parametry modelu
a = 1.2;  % birth rate of prey
b = 1.0;  % prey death rate due to predation
c = 0.3;  % predator birth rate from prey consumption
d = 0.8;  % predator death rate

% Warunki początkowe
x_0 = 1;   % Początkowa populacja ofiar
y_0 = 2;   % Początkowa populacja drapieżników

% Parametry symulacji
T = 30;          % Czas symulacji
dt = 0.0001;      % Krok czasowy
N = T/dt;        % Liczba kroków

% Inicjalizacja wektorów
t = linspace(0, T, N); % Czas
x = zeros(1, N);       % Populacja ofiar
y = zeros(1, N);       % Populacja drapieżników

% Ustawienie wartości początkowych
x(1) = x_0;
y(1) = y_0;

for i = 1:N-1
    dx = (a - b*y(i)) * x(i);
    dy = (c * x(i) - d) * y(i);
    
    x(i+1) = x(i) + dx * dt;
    y(i+1) = y(i) + dy * dt;
end

figure;
    plot(t, x, 'b', 'LineWidth', 1.5); hold on;
    plot(t, y, 'r', 'LineWidth', 1.5);
    xlabel('Czas');
    ylabel('Populacja');
    legend('Ofiary', 'Drapieżnicy');
    title('Model Lotki-Volterry');
    grid on;