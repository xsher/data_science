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
