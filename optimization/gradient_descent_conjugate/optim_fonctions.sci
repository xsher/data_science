function [J,G]=cost1(v);
    n = length(v)
    J = 0
    G = []
    for i=1:n
        J = J + (v(i) - 1)^2
        gradient = 2*(v(i)-1)
        G = cat(1, G, gradient)
    end
endfunction


function [J,G]=cost2(v);
    n = length(v)
    J = 0
    G = []
    for i=1:n
        J = J + (v(i) - i)^2
        gradient = 2*(v(i)-i)
        G = cat(1, G, gradient)
    end 
endfunction


function [J,G]=costR(v);
    n = length(v)
    J = 0
    G = []
    for i=1:(n-1)
        J = J + ((v(i+1) - v(i)^2)^2 + (v(i)- 1)^2)
        if i == 1 then
            gradient = 0 // padding for i=1
        elseif i > 1 then
            gradient = 2*(v(i)-v(i-1)^2)-(4*v(i)*(v(i+1) - v(i)^2)) + (2*(v(i)-1))
        end
        G = cat(1, G, gradient)
    end
    G = cat(1, G, 0) // padding when i=N
endfunction


function [J, G]=costH(v)
    x = v(1)
    y = v(2)
    
    J = (x^2 + y -2)^2 + (y^2 - 2*x +1)^2
    G(1) = (4*x*(x^2+y-2)) - (4*(y^2-2*x+1))
    G(2) = (2*(x^2+y-2)) + (4*y*(y^2-2*x+1))
endfunction


function Avk=Av(v)
    // A = tridiag[-1,2,-1]
    n = length(v)
    Avk(1) = 2*v(1) - v(2)
    for i=2:(n-1)
        Avk(i) = -v(i-1) + 2*v(i) - v(i+1)
    end
    Avk(n) = -v(n-1) + 2*v(n)
endfunction

function A1k=Av1(v)
    // A = tridiag[-1,2,-1]
    n = length(v)
    A = eye(n)
    A1k = 2 * eye(n) * v
endfunction

function Bvk=Bv(v)
    n = length(v)
    Bvk(1)  = 4*v(1) - v(2) - v(3)
    Bvk(2)  = -v(1) + 4*v(2) - v(3) - v(4)
    for i=3:(n-2)
        Bvk(i) = -v(i-2) -v(i-1) + 4*v(i) - v(i+1) - v(i+2)
    end
    Bvk(n-1) = -v(n-3) - v(n-2) + 4*v(n-1) - v(n)
    Bvk(n)   = -v(n-2) - v(n-1) + 4*v(n)
endfunction


function [J, G]=cost3(v)
    f(1:length(v)) = 1
    J = 1/2 * Av(v)' * v - f' * v
    G = Av(v) - f
endfunction


function [J, G]=cost4(v)
    f(1:length(v)) = 1
    J = 1/2 * Bv(v)' * v - f' * v
    G = Bv(v) - f
endfunction

function [J, G]=costEps(v, eps)
    n = length(v)
    J = 0
    G = []
    sum(v)
    for i=1:(n-1)
        J = J + (v(i) + v(i+1) - (n/2))^2
        if i == 1 then
            gradient = 2 * (v(1) + v(2) - n/2) // padding for i=1
        elseif i > 1 then
            gradient = 4 * v(i) + 2*v(i-1) + 2*v(i+1) - 2*n
        end
        G = cat(1, G, gradient)
    end

    G(n) =  2 * (v(n-1) + v(n) - n/2)
    G = (1/eps) * (G + 2*v)
    J = (1/eps * J) + sum((v)^2)
endfunction
