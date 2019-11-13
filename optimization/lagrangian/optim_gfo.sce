// load the cost functions
exec('/Users/sherly/Documents/EIT-UNS/uns/S2P1/optim_fonctions.sci', -1)

function [u1, couts]=pasBB1(EPSG, kmax, rho0, u0, cost)
    if cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "cost5" then
        cost_fn = cost5
    elseif cost == "cost6" then
        cost_fn = cost6
    elseif cost == "costR" then
        cost_fn = costR
    end

    // we first generate u1 with the gradient a pas fixe
    [j0, g0] = cost_fn(u0)
    u1 = u0 - rho0 * g0

    for k = 1:kmax
        [j1, g1] = cost_fn(u1)
        couts(k) = j1
        if (norm(g1) < EPSG) then
            disp("Achieved convergence at iteration: ")
            disp(k)
            break;
        else
            skmin1 = u1 - u0;
            ykmin1 = g1 - g0;
            rhok = (skmin1' * skmin1) / (ykmin1' * skmin1)
            u2 = u1 - rhok * g1;
            u0 = u1;
            u1 = u2;
            g0 = g1;
        end
    end
    if k == kmax then
        disp("Did not achieve convergence at max iteration")
    end 

endfunction


function [u1, couts]=pasBB2(EPSG, kmax, rho0, u0, cost)
    if cost == "cost1" then
        cost_fn = cost1
    elseif cost == "cost2" then
        cost_fn = cost2
    elseif cost == "cost5" then
        cost_fn = cost5
    elseif cost == "cost6" then
        cost_fn = cost6
    elseif cost == "costR" then
        cost_fn = costR
    end

    // we first generate u1 with the gradient a pas fixe
    [j0, g0] = cost_fn(u0)
    u1 = u0 - rho0 * g0

    for k = 1:kmax
        [j1, g1] = cost_fn(u1)
        couts(k) = j1
        if (norm(g1) < EPSG) then
            disp("Achieved convergence at iteration: ")
            disp(k)
            break;
        else
            skmin1 = u1 - u0;
            ykmin1 = g1 - g0;
            rhok = (skmin1' * ykmin1) / (ykmin1' * ykmin1)
            u2 = u1 - rhok * g1;
            u0 = u1;
            u1 = u2;
            g0 = g1;
        end
    end
    if k == kmax then
        disp("Did not achieve convergence at max iteration")
    end 

endfunction

function []=plotfn(BB1cost5, BB2cost5, BB1cost6, BB2cost6, BB1costR, BB2costR)
    plot(BB1cost5, "-r-");
    plot(BB2cost5, "-b-");
    plot(BB1cost6, "or-");
    plot(BB2cost6, "ob-");
    plot(BB1costR, "*r-");
    plot(BB2costR, "*b-");
    xtitle ( "Curves of convergence of cost" , "Iterations" , "Cost" );
    legend ( "Cost of BB1 on J5" , "Cost of BB2 on J5", "Cost of BB1 on J6" , "Cost of BB2 on J6",    "Cost of BB1 on JR" , "Cost of BB2 on JR" );
endfunction



    
 
