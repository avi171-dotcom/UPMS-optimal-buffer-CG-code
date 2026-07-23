
import itertools
import time
import numpy as np
from scipy. optimize import milp, LinearConstraint, Bounds



def generate_instance(n_jobs=3, n_stations=3, seed=42):
    rng = np.random.default_rng(seed)
    PT = rng.integers(1, 16, size=(n_jobs, n_stations))         
    ST = rng.integers(1, 6, size=(n_jobs, n_stations))           
    W = rng.integers(1, 6, size=(n_jobs, n_stations))           
    n_resources = 2
    Avl = rng.integers(1, 6, size=(n_stations, n_resources))     
    mp = rng.integers(2, 6, size=n_stations)                     

    inst = dict(
        n_jobs=n_jobs, n_stations=n_stations, n_resources=n_resources,
        PT=PT, ST=ST, W=W, Avl=Avl, mp=mp,
        jobs=[f"J{j+1}" for j in range(n_jobs)],
        stations=[f"M{k+1}" for k in range(n_stations)],
        resources=[f"R{r+1}" for r in range(n_resources)],
        # buffer capacity between consecutive stations (space units)
        buffer_capacity=[int(rng.integers(6, 12)) for _ in range(n_stations - 1)],
        seed=seed,
    )
    return inst


# --------------------------------------------------------------------------

def simulate_blocking(perm, inst):
    n, m = inst['n_jobs'], inst['n_stations']
    PT, ST = inst['PT'], inst['ST']
    t = np.zeros((n, m))          # start of processing (after setup)
    dep = np.zeros((n, m))        # time job departs the machine (leaves the machine)
    finish_proc = np.zeros((n, m))  # time raw processing (setup+proc) ends

    for i in range(n):
        j = perm[i]
        for k in range(m):
            # a job's own readiness for stage k = when IT finished stage k-1
            arrival = 0 if k == 0 else finish_proc[i, k - 1]
            if i == 0:
                machine_free = 0
            elif k == m - 1:
                # last machine: no downstream blocking, freed once previous
                # job finished processing there
                machine_free = finish_proc[i - 1, k]
            else:
                # machine k freed only once job i-1 STARTS at machine k+1
                machine_free = t[i - 1, k + 1]
            start_setup = max(arrival, machine_free)
            t[i, k] = start_setup
            finish_proc[i, k] = start_setup + ST[j, k] + PT[j, k]
        # departure time of job i from each machine (needed for blocking viz)
        for k in range(m - 1):
            dep[i, k] = t[i, k + 1]
        dep[i, m - 1] = finish_proc[i, m - 1]

    makespan = finish_proc[n - 1, m - 1]
    blocking = np.zeros((n, m))
    for i in range(n):
        for k in range(m - 1):
            blocking[i, k] = max(0.0, dep[i, k] - finish_proc[i, k])
    return dict(perm=perm, start=t, finish=finish_proc, depart=dep,
                makespan=makespan, blocking=blocking, mode='no_buffer')


def simulate_buffer(perm, inst):
    n, m = inst['n_jobs'], inst['n_stations']
    PT, ST = inst['PT'], inst['ST']
    t = np.zeros((n, m))
    finish_proc = np.zeros((n, m))

    for i in range(n):
        j = perm[i]
        for k in range(m):
            arrival = 0 if k == 0 else finish_proc[i, k - 1]
            machine_free = 0 if i == 0 else finish_proc[i - 1, k]
            start_setup = max(arrival, machine_free)
            t[i, k] = start_setup
            finish_proc[i, k] = start_setup + ST[j, k] + PT[j, k]

    makespan = finish_proc[n - 1, m - 1]
    # buffer occupancy: job i sits in buffer k in [finish_proc[i,k], t[i,k+1]]
    buffer_intervals = []
    for i in range(n):
        for k in range(m - 1):
            s, e = finish_proc[i, k], t[i, k + 1]
            if e > s:
                buffer_intervals.append((perm[i], k, s, e))
    return dict(perm=perm, start=t, finish=finish_proc, depart=finish_proc.copy(),
                makespan=makespan, blocking=np.zeros((n, m)),
                buffer_intervals=buffer_intervals, mode='buffer')



# 3. MILP  

def _var_index(n, m):
    idx_x = lambda j, i: j * n + i
    idx_t = lambda i, k: n * n + i * m + k
    idx_c = n * n + n * m
    n_vars = n * n + n * m + 1
    return idx_x, idx_t, idx_c, n_vars


def build_and_solve_milp(inst, buffer=False, time_limit=30):
    
    n, m = inst['n_jobs'], inst['n_stations']
    PT, ST = inst['PT'], inst['ST']
    idx_x, idx_t, idx_c, n_vars = _var_index(n, m)

    rows, lows, ups = [], [], []

    def add_row(coeffs, lo, up):
        row = np.zeros(n_vars)
        for idx, val in coeffs.items():
            row[idx] += val
        rows.append(row); lows.append(lo); ups.append(up)

    
    for j in range(n):
        add_row({idx_x(j, i): 1 for i in range(n)}, 1, 1)
    
    for i in range(n):
        add_row({idx_x(j, i): 1 for j in range(n)}, 1, 1)

    
    for i in range(n):
        for k in range(1, m):
            coeffs = {idx_t(i, k): 1, idx_t(i, k - 1): -1}
            for j in range(n):
                coeffs[idx_x(j, i)] = coeffs.get(idx_x(j, i), 0) - (ST[j, k - 1] + PT[j, k - 1])
            add_row(coeffs, 0, np.inf)

    
    for i in range(1, n):
        for k in range(m):
            if not buffer and k < m - 1:
                # blocking: freed when previous job STARTS at k+1
                add_row({idx_t(i, k): 1, idx_t(i - 1, k + 1): -1}, 0, np.inf)
            else:
                
                coeffs = {idx_t(i, k): 1, idx_t(i - 1, k): -1}
                for j in range(n):
                    coeffs[idx_x(j, i - 1)] = coeffs.get(idx_x(j, i - 1), 0) - (ST[j, k] + PT[j, k])
                add_row(coeffs, 0, np.inf)

    
    add_row({idx_t(0, 0): 1}, 0, 0)

   
    for i in range(n):
        coeffs = {idx_c: 1, idx_t(i, m - 1): -1}
        for j in range(n):
            coeffs[idx_x(j, i)] = coeffs.get(idx_x(j, i), 0) - (ST[j, m - 1] + PT[j, m - 1])
        add_row(coeffs, 0, np.inf)

    A = np.array(rows)
    constraints = LinearConstraint(A, lows, ups)

    c = np.zeros(n_vars)
    c[idx_c] = 1.0

    integrality = np.zeros(n_vars)
    integrality[: n * n] = 1  # x variables binary
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    ub[: n * n] = 1
    bounds = Bounds(lb, ub)

    t0 = time.perf_counter()
    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=bounds, options={'time_limit': time_limit})
    cpu = time.perf_counter() - t0

    perm = []
    xmat = res.x[: n * n].reshape(n, n)  # xmat[j,i]
    for i in range(n):
        j = int(np.argmax(xmat[:, i]))
        perm.append(j)

    sim = simulate_buffer(perm, inst) if buffer else simulate_blocking(perm, inst)
    return dict(perm=perm, makespan=res.fun, cpu_time=cpu, status=res.status,
                message=res.message, sim=sim, n_vars=n_vars, n_constraints=A.shape[0])



# 4. COLUMN GENERATION  

def _cost(perm, inst, buffer):
    return (simulate_buffer(perm, inst) if buffer else simulate_blocking(perm, inst))['makespan']


def solve_rmp_lp(columns, costs, n):
    
    n_cols = len(columns)
    n_vars = n_cols
    # equality constraints: for every job j, sum_p lambda_p*[job j used]  = 1
    #                        for every pos i, sum_p lambda_p*[pos i used] = 1
    #                        sum lambda_p = 1 (convexity)
    A_rows = []
    for j in range(n):
        A_rows.append([1.0 if j in col else 0.0 for col in columns])
    for i in range(n):
        A_rows.append([1.0 for _ in columns])  # every column uses every position exactly once (full perm)
    A_rows.append([1.0 for _ in columns])       # convexity
    A = np.array(A_rows)
    b = np.array([1.0] * n + [1.0] * n + [1.0])

    c = np.array(costs)
    integrality = np.zeros(n_vars)
    bounds = Bounds(np.zeros(n_vars), np.ones(n_vars))
    constraints = LinearConstraint(A, b, b)
    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    return res


def column_generation(inst, buffer=False, max_iter=25, verbose=False):
    
    n = inst['n_jobs']
    all_perms = list(itertools.permutations(range(n)))

    # seed RMP with 2 columns: natural order & reverse order
    columns = [tuple(range(n)), tuple(reversed(range(n)))]
    costs = [_cost(list(p), inst, buffer) for p in columns]

    history = []
    t0 = time.perf_counter()
    it = 0
    while it < max_iter:
        it += 1
        # ---- solve LP relaxation of RMP (relax integrality to get duals) ----
        n_cols = len(columns)
        A_rows = []
        for j in range(n):
            A_rows.append([1.0] * n_cols)          
        for i in range(n):
            A_rows.append([1.0] * n_cols)         
        A_rows.append([1.0] * n_cols)              
        A = np.array(A_rows)
        b = np.array([1.0] * n + [1.0] * n + [1.0])
        c = np.array(costs)
        bounds = Bounds(np.zeros(n_cols), np.ones(n_cols))
        constraints = LinearConstraint(A, b, b)
        res_lp = milp(c=c, constraints=constraints,
                       integrality=np.zeros(n_cols), bounds=bounds)

       
        best_val = res_lp.fun if res_lp.status == 0 else min(costs)

        # ---- pricing problem: find column with cost < best_val not yet in pool
        candidates = [p for p in all_perms if p not in columns]
        if not candidates:
            history.append(dict(iter=it, rmp_obj=best_val, new_col=None, reduced_cost=0.0))
            break
        cand_costs = [(_cost(list(p), inst, buffer), p) for p in candidates]
        cand_costs.sort(key=lambda x: x[0])
        best_cand_cost, best_cand = cand_costs[0]
        reduced_cost = best_cand_cost - best_val

        history.append(dict(iter=it, rmp_obj=best_val, new_col=best_cand,
                             reduced_cost=reduced_cost))
        if verbose:
            print(f"  iter {it}: RMP={best_val:.2f}  candidate={best_cand} "
                  f"cost={best_cand_cost:.2f}  reduced_cost={reduced_cost:.2f}")

        if reduced_cost >= -1e-6:
            break
        columns.append(best_cand)
        costs.append(best_cand_cost)

    # ---- final INTEGER restricted master problem ----
    n_cols = len(columns)
    A_rows = []
    for j in range(n):
        A_rows.append([1.0] * n_cols)
    for i in range(n):
        A_rows.append([1.0] * n_cols)
    A_rows.append([1.0] * n_cols)
    A = np.array(A_rows)
    b = np.array([1.0] * n + [1.0] * n + [1.0])
    c = np.array(costs)
    bounds = Bounds(np.zeros(n_cols), np.ones(n_cols))
    constraints = LinearConstraint(A, b, b)
    res_int = milp(c=c, constraints=constraints,
                    integrality=np.ones(n_cols), bounds=bounds)
    cpu = time.perf_counter() - t0

    best_idx = int(np.argmax(res_int.x)) if res_int.x is not None else int(np.argmin(costs))
    best_perm = list(columns[best_idx])
    sim = simulate_buffer(best_perm, inst) if buffer else simulate_blocking(best_perm, inst)

    return dict(perm=best_perm, makespan=costs[best_idx], cpu_time=cpu,
                iterations=it, history=history, columns=columns, costs=costs,
                sim=sim)



def compute_metrics(sim, inst):
    n, m = inst['n_jobs'], inst['n_stations']
    PT, ST = inst['PT'], inst['ST']
    perm = sim['perm']
    makespan = sim['makespan']

    busy = np.zeros(m)
    for i in range(n):
        j = perm[i]
        for k in range(m):
            busy[k] += ST[j, k] + PT[j, k]

    blocking_time = sim['blocking'].sum(axis=0) if sim['mode'] == 'no_buffer' else np.zeros(m)
    idle_time = makespan - busy - blocking_time
    idle_time = np.clip(idle_time, 0, None)
    utilization = busy / makespan

    buffer_occ_total = np.zeros(max(m - 1, 1))
    buffer_peak = np.zeros(max(m - 1, 1))
    if sim['mode'] == 'buffer' and sim.get('buffer_intervals'):
        for (_, k, s, e) in sim['buffer_intervals']:
            buffer_occ_total[k] += (e - s)
        # peak simultaneous occupancy per buffer (event sweep)
        for k in range(m - 1):
            events = []
            for (_, kk, s, e) in sim['buffer_intervals']:
                if kk == k:
                    events.append((s, 1)); events.append((e, -1))
            events.sort()
            cur = peak = 0
            for _, delta in events:
                cur += delta
                peak = max(peak, cur)
            buffer_peak[k] = peak

    total_waiting = buffer_occ_total.sum()  # time jobs spend waiting in buffer
    total_blocking = blocking_time.sum()

    return dict(
        machine_busy=busy, machine_idle=idle_time, machine_utilization=utilization,
        blocking_time_per_machine=blocking_time, total_blocking_time=total_blocking,
        buffer_occupied_time=buffer_occ_total, buffer_peak=buffer_peak,
        total_waiting_time=total_waiting, makespan=makespan,
    )
