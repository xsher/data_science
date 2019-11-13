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
            gradient = 4 * v(i) * (v(i+1) - v(i)^2) + 2 * (v(i) - 1)
        elseif i > 1 then
            gradient = 2*(v(i)-v(i-1)^2)-(4*v(i)*(v(i+1) - v(i)^2)) + (2*(v(i)-1))
        end
        G = cat(1, G, gradient)
    end
    Gn = 2*(v(n) - 1)
    G = cat(1, G, Gn)
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

function [J, G]=cost5(v)
    f(1:length(v)) = 1
    J = 1/2 * Av(v)' * v - f' * v + sum(v^2)
    G = Av(v) - f + 2 * v
endfunction

function [J, G]=cost6(v)
    f(1:length(v)) = 1
    J = 1/2 * Av(v)' * v - f' * v + sum(v^4)
    G = Av(v) - f  + 4 * v^3
endfunction
