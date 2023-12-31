import cma
import numpy as np
from solution import solution
import time


# opts = cma.CMAOptions()
# opts.set("bounds", [0, 1])
# opts.set("popsize", 12)
# opts.set("maxiter", 400)

# es = cma.CMAEvolutionStrategy(np.random.rand(5), 0.5)
# es.optimize(lambda x: sum(x**2))  
# s = es.result
# xbesttt= es.result.xbest
# cmaes_time = es.timer.elapsed
# convergence_curve = es.fit.histbest
# convergence_curve.reverse()
# print("helo")
def CMAES(objf, lb, ub, dim, PopSize, iters):

    convergence_curve = np.zeros(iters)

    opts = cma.CMAOptions()
    opts.set("bounds", [lb, ub])
    opts.set("popsize", PopSize)
    opts.set("maxiter", iters)
    opts.set("tolx", 1e-20)

    # solution
    s           = solution()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    seed        = np.random.uniform(0, 1, dim) * (ub - lb) + lb
    es          = cma.CMAEvolutionStrategy(seed, 0.5, opts)
    es.optimize(objf)
    
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    
    s.best = es.result.fbest; s.bestIndividual = es.result.xbest

    cmaes_time = es.timer.elapsed
    s.executionTime = cmaes_time

    histbest = es.fit.histbest
    histbest.reverse()

    convergence_curve[0:len(histbest)] = histbest
    s.convergence = convergence_curve

    s.lb = lb; s.ub = ub; s.dim = dim; s.popnum = PopSize
    s.optimizer = "CMAES"; s.objfname = objf.__name__


    # return solution
    return s
