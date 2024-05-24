function y = func(x)    
x1=x(1);
    x2=x(2);

       sum1 = x1^2 + x2^2;
   prod1 = cos(x1/sqrt(1)) * cos(x2/sqrt(2));

   y = 1 + (1/4000) * sum1 - prod1;
end