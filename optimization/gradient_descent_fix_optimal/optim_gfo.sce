// load the cost functions
exec('/Users/sherly/Documents/EIT-UNS/uns/S2P1/optim_fonctions.sci', -1)

function [u0, couts]=pasfixe(EPSG, kmax, rho0, u0, cost)
    if cost == "costR" then
        cost_fn = costR
    elseif cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "costH" then
        cost_fn = costH
    end

    try
        for k = 1:kmax
            [j, g] = cost_fn(u0)
            couts(k) = j
            if (norm(g) < EPSG) then
                disp("Achieved convergence at iteration: ")
                disp(k-1)
                break;
            else
                u1 = u0 - rho0 * g;
                u0 = u1;
            end
        end
    catch
        disp("Invalid cost or gradient")
        disp(j)
        disp(g)
        disp("Unable to reach convergence")
        disp(k)
    end

    if k == kmax then
        disp("Did not achieve convergence at max iteration")
    end 

endfunction


function [u0, couts]=pasoptimal(EPSG, kmax, rho0, u0, cost)
    if cost == "costR" then
        cost_fn = costR
    elseif cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "costH" then
        cost_fn = costH
    end
    
    [Jk, Gk] = cost_fn(u0)
    uk = u0 - rho0 * Gk
    [Jk, Gk] = cost_fn(uk)

    for k = 1:kmax
        [j, g] = cost_fn(u0)
        couts(k) = j
        disp("got here")
        if (norm(g) < EPSG) then
            disp("Achieved convergence at iteration: ")
            disp(k-1)
            break;
        else
            disp(k)
            c = j
            b = -(norm(g))^2
            uk = u0 - rho0 * g
            [Jk, Gk] = cost_fn(uk)
            a = (Jk - b*rho0 - c) / (rho0)^2
            rho0 = -b/(2*a)
            u1 = u0 - rho0 * g;
            u0 = u1;
        end
    end

    if k == kmax then
        disp("Did not achieve convergence at max iteration")
    end
    
endfunction


function []=plotfn(coutsGF, coutsG0)
    plot(coutsGF, "-r");
    plot(coutsGO, "-b");
    xtitle ( "Curves of convergence of cost" , "Iterations" , "Cost" );
    legend ( "Cost of GF" , "Cost of GO" );
endfunction
