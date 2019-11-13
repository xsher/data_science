// load the cost functions
exec('/Users/sherly/Documents/EIT-UNS/uns/S2P1/optim_fonctions.sci', -1)

function [u0, couts]=pasfixe(EPSG, kmax, rho0, u0, cost)
    if cost == "costR" then
        cost_fn = costR
    elseif cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "cost3" then
        cost_fn = cost3
    elseif cost == "cost4" then
        cost_fn = cost4
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



function [u0, couts]=conjuguee(EPSG, kmax, u0, cost)
    if cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "cost3" then
        cost_fn = cost3
    elseif cost == "cost4" then
        cost_fn = cost4
    end

    
    if cost == "cost1" then
        Amat = Av1
    elseif cost == "cost2" then
        Amat = Av1
    elseif cost == "cost3" then
        Amat = Av
    elseif cost == "cost4" then
        Amat = Bv
    end

    // set d0 = G0
    [j0, g0] = cost_fn(u0)
    d0 = g0
    couts = 0


    for k = 1:kmax
        [jk, gk] = cost_fn(u0)
        couts(k) = jk
        if (norm(gk) < EPSG) then
            disp("Achieved convergence at iteration: ")
            disp(k-1)
            break;
        else
            // rhok = <gk, dk> / <Adk, dk>
            denominator = Amat(d0)' * d0;
            rhok = (gk' * d0) / denominator;
            u1 = u0 - rhok * d0;
            [j1, g1] = cost_fn(u1);
            betak = - (g1' * Amat(d0)) / (d0' * Amat(d0));
            d1 = g1 + betak * d0;
            u0 = u1;
            d0 = d1;
        end
    end
//    catch
//        disp("Invalid cost or gradient")
//        disp(jk)
//        disp(gk)
//        disp("Unable to reach convergence")
//        disp(k)
//        disp(couts)
//    end

    if k == kmax then
        disp("Did not achieve convergence at max iteration")
    end 
endfunction

function [u0, couts]=conjuguee_fletcher(EPSG, kmax, rho0, u0, cost, p)
    if cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "cost3" then
        cost_fn = cost3
    elseif cost == "cost4" then
        cost_fn = cost4
    elseif cost == "costEps" then
        cost_fn = costEps
    end
    eps = 10^(-p)

    // set d0 = G0
    [j0, g0] = cost_fn(u0, eps)
    d0 = g0
    couts = 0

    try
        for k = 1:kmax
            [jk, gk] = cost_fn(u0, eps)
            couts(k) = jk
            if (norm(gk) < EPSG) then
                disp("Achieved convergence at iteration: ")
                disp(k-1)
                break;
            else
                u1 = u0 - rho0 * d0;
                [j1, g1] = cost_fn(u1, eps);
                betak =  (g1' * g1) / (gk' * gk);
                d1 = g1 + betak * d0;
                u0 = u1;
                d0 = d1;
            end
        end
    catch
        disp("Invalid cost or gradient")
        disp(jk)
        disp(gk)
        disp("Unable to reach convergence")
        disp(k)
        disp(couts)
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
