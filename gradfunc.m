function ypr = gradfunc(x)
   syms x1 x2;

   % Griewank fonksiyonunu tanımla
   sum1 = x1^2 + x2^2;
   prod1 = cos(x1/sqrt(1)) * cos(x2/sqrt(2));

   F = 1 + (1/4000) * sum1 - prod1;

   % Gradyanı hesapla
   g = gradient(F);
   ypr = double(subs(g, [x1, x2], [x(1), x(2)]));
end