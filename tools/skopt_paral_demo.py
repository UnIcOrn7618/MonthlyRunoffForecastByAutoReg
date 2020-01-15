from skopt import Optimizer
from skopt.space import Real
from joblib import Parallel, delayed
# example objective taken from skopt
from skopt.benchmarks import branin
print(branin)

optimizer = Optimizer(
    dimensions=[Real(-5.0, 10.0), Real(0.0, 15.0)],
    random_state=1,
    base_estimator='gp'
)

for i in range(10): 
    x = optimizer.ask(n_points=4)  # x is a list of n_points points    
    y = Parallel(n_jobs=4)(delayed(branin)(v) for v in x)  # evaluate points in parallel
    optimizer.tell(x, y)

# takes ~ 20 sec to get here
print(min(optimizer.yi))