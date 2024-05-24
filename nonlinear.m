clear all
close all
clc

X1 = -100:1:100;
X2 = -100:1:100;
[x1, x2] = meshgrid(X1, X2);

sum1 = x1.^2 + x2.^2;
prod1 = cos(x1./sqrt(1)) .* cos(x2./sqrt(2));

F = 1 + (1/4000) * sum1 - prod1;
realFMin = min(min(F));
mesh(x1, x2, F)

figure
contourf(x1, x2, F)
hold on

% Generate three random initial points to be used by all algorithms
x0s = rand(2, 3) * 200 - 100; % [-100, 100] aralığında rastgele başlatma

% Function, gradient, and Hessian
func = @(x) 1 + (1/4000)*(x(1)^2 + x(2)^2) - cos(x(1)/sqrt(1)) * cos(x(2)/sqrt(2));
gradfunc = @(x) [(1/2000)*x(1) + sin(x(1)/sqrt(1)) * cos(x(2)/sqrt(2)) / sqrt(1);
                 (1/2000)*x(2) + cos(x(1)/sqrt(1)) * sin(x(2)/sqrt(2)) / sqrt(2)];
hessianfunc = @(x) [(1/2000) + (1/1)*cos(x(1)/sqrt(1))*cos(x(2)/sqrt(2)), sin(x(1)/sqrt(1))*sin(x(2)/sqrt(2))/sqrt(2);
                    sin(x(1)/sqrt(1))*sin(x(2)/sqrt(2))/sqrt(2), (1/2000) + (1/2)*cos(x(1)/sqrt(1))*cos(x(2)/sqrt(2))];

%% Newton-Raphson
fprintf('Newton-Raphson Algorithm\n');
epsilon = 10^(-4);

for i = 1:3
    x0 = x0s(:, i);
    x = x0;
    
    tic;
    fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x))
    plot(x(1), x(2), 'r.')
    x_next = x - inv(hessianfunc(x)) * gradfunc(x);
    fprintf('k=2, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
    plot(x_next(1), x_next(2), 'r*')
    k = 3;
    while norm(gradfunc(x_next)) > epsilon
        x = x_next;
        x_next = x - inv(hessianfunc(x)) * gradfunc(x);
        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
        plot(x_next(1), x_next(2), 'r*')
        k = k + 1;
    end
    execution_time = toc;
    fprintf('Execution time: %f seconds\n', execution_time);
    fprintf('Optimal value: %f, Optimal point: (%f, %f)\n', func(x_next), x_next(1), x_next(2));
end

title('Newton-Raphson Algorithm')
set(gca, 'fontsize', 35)
set(findobj(gca, 'Type', 'Line', 'Linestyle', '--'), 'LineWidth', 2);

%% Hestenes-Stiefel Algorithm
figure
contourf(x1, x2, F)
hold on

fprintf('Hestenes-Stiefel Algorithm\n');
epsilon = 10^(-4);

for i = 1:3
    x0 = x0s(:, i);
    x = x0;

    tic;
    fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x))
    plot(x(1), x(2), 'r.')
    g = gradfunc(x);
    d = -g;

    % alpha argmin procedure
    alpha = 0:0.01:1;  
    funcalpha = zeros(length(alpha), 1);

    for j = 1:length(alpha)
        funcalpha(j) = func(x + alpha(j) * d);
    end
    [~, ind] = min(funcalpha);
    alpha = alpha(ind);

    x_next = x + alpha * d;
    g_next = gradfunc(x_next);
    beta = (g_next' * (g_next - g)) / (d' * (g_next - g));
    d_next = -g_next + beta * d;

    fprintf('k=2, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
    plot(x_next(1), x_next(2), 'r*')
    k = 3;
    while norm(gradfunc(x_next)) > epsilon
        x = x_next;
        g = g_next;
        d = d_next;

        % alpha argmin procedure
        alpha = 0:0.01:1;
        funcalpha = zeros(length(alpha), 1);
        for j = 1:length(alpha)
            funcalpha(j) = func(x + alpha(j) * d);
        end
        [~, ind] = min(funcalpha);
        alpha = alpha(ind);

        x_next = x + alpha * d;
        g_next = gradfunc(x_next);
        beta = (g_next' * (g_next - g)) / (d' * (g_next - g));
        d_next = -g_next + beta * d;

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
        plot(x_next(1), x_next(2), 'r*')
        k = k + 1;
    end
    execution_time = toc;
    fprintf('Execution time: %f seconds\n', execution_time);
    fprintf('Optimal value: %f, Optimal point: (%f, %f)\n', func(x_next), x_next(1), x_next(2));
end

title('Hestenes-Stiefel Algorithm')
set(gca, 'fontsize', 35)
set(findobj(gca, 'Type', 'Line', 'Linestyle', '--'), 'LineWidth', 2);

%% Polak-Ribiére Algorithm
figure
contourf(x1, x2, F)
hold on

fprintf('Polak-Ribiére Algorithm\n');
epsilon = 10^(-4);

for i = 1:3
    x0 = x0s(:, i);
    x = x0;

    tic;
    fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x))
    plot(x(1), x(2), 'r.')
    g = gradfunc(x);
    d = -g;

    % alpha argmin procedure
    alpha = 0:0.01:1;  
    funcalpha = zeros(length(alpha), 1);

    for j = 1:length(alpha)
        funcalpha(j) = func(x + alpha(j) * d);
    end
    [val, ind] = min(funcalpha);
    alpha = alpha(ind);

    x_next = x + alpha * d;
    g_next = gradfunc(x_next);
    beta = (g_next' * (g_next - g)) / (g' * g);
    d_next = -g_next + beta * d;

    fprintf('k=2, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
    plot(x_next(1), x_next(2), 'r*')
    k = 3;
    while norm(gradfunc(x_next)) > epsilon
        x = x_next;
        g = g_next;
        d = d_next;

        % alpha argmin procedure
        alpha = 0:0.01:1;
        funcalpha = zeros(length(alpha), 1);
        for j = 1:length(alpha)
            funcalpha(j) = func(x + alpha(j) * d);
        end
        [val, ind] = min(funcalpha);
        alpha = alpha(ind);

        x_next = x + alpha * d;
        g_next = gradfunc(x_next);
        beta = (g_next' * (g_next - g)) / (g' * g);
        d_next = -g_next + beta * d;

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
        plot(x_next(1), x_next(2), 'r*')
        k = k + 1;
    end
    execution_time = toc;
    fprintf('Execution time: %f seconds\n', execution_time);
    fprintf('Optimal value: %f, Optimal point: (%f, %f)\n', func(x_next), x_next(1), x_next(2));
end

title('Polak-Ribiére Algorithm')
set(gca, 'fontsize', 25)
set(findobj(gca, 'Type', 'Line', 'Linestyle', '--'), 'LineWidth', 2);

%% Fletcher-Reeves Algorithm
figure
contourf(x1, x2, F)
hold on

fprintf('Fletcher-Reeves Algorithm\n');
epsilon = 10^(-4);

for i = 1:3
    x0 = x0s(:, i);
    x = x0;

    tic;
    fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x))
    plot(x(1), x(2), 'r.')
    g = gradfunc(x);
    d = -g;

    % alpha argmin procedure
    alpha = 0:0.01:1;  
    funcalpha = zeros(length(alpha), 1);

    for j = 1:length(alpha)
        funcalpha(j) = func(x + alpha(j) * d);
    end
    [val, ind] = min(funcalpha);
    alpha = alpha(ind);

    x_next = x + alpha * d;
    g_next = gradfunc(x_next);
    beta = (g_next' * g_next) / (g' * g);
    d_next = -g_next + beta * d;

    fprintf('k=2, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
    plot(x_next(1), x_next(2), 'r*')
    k = 3;
    while norm(gradfunc(x_next)) > epsilon
        x = x_next;
        g = g_next;
        d = d_next;

        % alpha argmin procedure
        alpha = 0:0.01:1;
        funcalpha = zeros(length(alpha), 1);
        for j = 1:length(alpha)
            funcalpha(j) = func(x + alpha(j) * d);
        end
        [val, ind] = min(funcalpha);
        alpha = alpha(ind);

        x_next = x + alpha * d;
        g_next = gradfunc(x_next);
        beta = (g_next' * g_next) / (g' * g);
        d_next = -g_next + beta * d;

        fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
        plot(x_next(1), x_next(2), 'r*')
        k = k + 1;
    end
    execution_time = toc;
    fprintf('Execution time: %f seconds\n', execution_time);
    fprintf('Optimal value: %f, Optimal point: (%f, %f)\n', func(x_next), x_next(1), x_next(2));
end

title('Fletcher-Reeves Algorithm')
set(gca, 'fontsize', 25)
set(findobj(gca, 'Type', 'Line', 'Linestyle', '--'), 'LineWidth', 2);

% %% Stochastic Gradient Descent Algorithm
% figure
% contourf(x1, x2, F)
% hold on
% 
% fprintf('Stochastic Gradient Descent Algorithm\n');
% epsilon = 10^(-4); % termination criterion
% 
% for i = 1:3
%     x0 = x0s(:, i);
%     x = x0;
% 
%     tic;
%     fprintf('k=1, x1=%f, x2=%f, f(x)=%f\n', x(1), x(2), func(x))
%     plot(x(1), x(2), 'r.')
%     g = gradfunc(x);
% 
%     % learning rate (you may need to tune this value)
%     alpha = 0.04;
% 
%     x_next = x - alpha * g; % gradient descent step
% 
%     fprintf('k=2, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
%     plot(x_next(1), x_next(2), 'r*')
%     k = 3;
%     while abs(func(x_next) - func(x)) > epsilon
%         x = x_next;
%         g = gradfunc(x); % compute gradient at new point
%         x_next = x - alpha * g; % gradient descent step
%         fprintf('k=%d, x1=%f, x2=%f, f(x)=%f, abs.error=%f\n', k, x_next(1), x_next(2), func(x_next), abs(func(x_next) - func(x)))
%         plot(x_next(1), x_next(2), 'r*')
%         k = k + 1;
%     end
%     execution_time = toc;
%     fprintf('Execution time: %f seconds\n', execution_time);
%     fprintf('Optimal value: %f, Optimal point: (%f, %f)\n', func(x_next), x_next(1), x_next(2));
% end
% 
% title('Stochastic Gradient Descent')
% set(gca, 'fontsize', 35)
% set(findobj(gca, 'Type', 'Line', 'Linestyle', '--'), 'LineWidth', 2);
