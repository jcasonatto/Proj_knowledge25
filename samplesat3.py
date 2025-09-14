
import random, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from sympy import And, Or, Not
from sympy.logic.boolalg import to_cnf as _sympy_to_cnf

Literal = Tuple[bool, object]  # (is_positive, sympy.Symbol)
Clause = List[Literal]

def _vars_set(F) -> Set[object]:
    return set(F.free_symbols)

def _random_assignment(V: Set[object], rng: Optional[random.Random]=None) -> Dict[object, bool]:
    rng = rng or random
    return {v: bool(rng.getrandbits(1)) for v in V}

def _eval_under(F, A: Dict[object, bool]) -> bool:
    return bool(F.subs(A))

def _cnf_clauses_force(F) -> List[Clause]:
    cnf = _sympy_to_cnf(F, simplify=True, force=True)
    if cnf == True:
        return []
    if cnf == False:
        return [[]]
    def lit(sym):
        if sym.func is Not:
            return (False, sym.args[0])
        return (True, sym)
    if cnf.func is Or:
        return [[lit(arg) for arg in cnf.args]]
    if cnf.func is And:
        return [[lit(l) for l in c.args] if c.func is Or else [lit(c)] for c in cnf.args]
    return [[lit(cnf)]]

def _cnf_clauses(F) -> List[Clause]:
    # Fast path when F already is CNF-like
    def _lit_from_expr(e):
        if e.func is Not and len(e.args) == 1 and e.args[0].is_Symbol:
            return (False, e.args[0])
        if e.is_Symbol:
            return (True, e)
        return None
    def clause_from(e):
        if e.func is Or:
            lits = []
            for a in e.args:
                lit = _lit_from_expr(a)
                if lit is None:
                    return None
                lits.append(lit)
            return lits
        else:
            lit = _lit_from_expr(e)
            if lit is not None:
                return [lit]
            return None
    if F == True:  return []
    if F == False: return [[]]
    if F.func is And:
        parsed = []
        for c in F.args:
            cl = clause_from(c)
            if cl is None:
                break
            parsed.append(cl)
        else:
            return parsed
    else:
        cl = clause_from(F)
        if cl is not None:
            return [cl]
    return _cnf_clauses_force(F)

def _clause_satisfied(clause: Clause, A: Dict[object, bool]) -> bool:
    for is_pos, var in clause:
        val = A[var]
        if (is_pos and val) or ((not is_pos) and (not val)):
            return True
    return False

def _unsat_clauses_idx(clauses: List[Clause], A: Dict[object, bool]) -> List[int]:
    return [i for i, c in enumerate(clauses) if not _clause_satisfied(c, A)]

def _breakcount(var, A: Dict[object, bool], clauses: List[Clause]) -> int:
    b = 0
    Av = A[var]
    for c in clauses:
        sat = False
        satisfied_by_other = False
        for is_pos, v in c:
            val = A[v]
            if (is_pos and val) or ((not is_pos) and (not val)):
                sat = True
                if v != var:
                    satisfied_by_other = True
        if sat and not satisfied_by_other:
            for is_pos, v in c:
                if v == var:
                    lit_is_true = (is_pos and Av) or ((not is_pos) and (not Av))
                    if lit_is_true:
                        b += 1
                    break
    return b

from dataclasses import dataclass
@dataclass
class SampleSATResult:
    sat: bool
    assignment: Dict[object, bool]
    tries_used: int
    flips_used: int
    last_unsat_count: int

def sample_sat(F,
               max_tries: int = 50,
               max_flips: int = 10000,
               noise: float = 0.5,
               rng: Optional[random.Random] = None) -> SampleSATResult:
    rng = rng or random.Random()
    V = _vars_set(F)
    clauses = _cnf_clauses(F)

    if clauses == []:
        A = _random_assignment(V, rng)
        return SampleSATResult(True, A, 0, 0, 0)
    if clauses == [[]]:
        return SampleSATResult(False, {}, 0, 0, len(clauses))

    for t in range(1, max_tries + 1):
        A = _random_assignment(V, rng)
        for f in range(1, max_flips + 1):
            if _eval_under(F, A):
                return SampleSATResult(True, dict(A), t, f, 0)
            unsat_idx = _unsat_clauses_idx(clauses, A)
            c = clauses[rng.choice(unsat_idx)]
            if rng.random() < noise:
                _, v = rng.choice(c)
                A[v] = not A[v]
            else:
                candidates = [v for _, v in c]
                bc = {v: _breakcount(v, A, clauses) for v in candidates}
                zeros = [v for v in candidates if bc[v] == 0]
                if zeros:
                    v_to_flip = rng.choice(zeros)
                else:
                    min_bc = min(bc.values())
                    best = [v for v in candidates if bc[v] == min_bc]
                    v_to_flip = rng.choice(best)
                A[v_to_flip] = not A[v_to_flip]
    last_unsat = len(_unsat_clauses_idx(clauses, A))
    return SampleSATResult(False, dict(A), max_tries, max_flips, last_unsat)

# --------- #SAT Estimation ---------
def _wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k/n
    denom = 1 + z**2/n
    center = (p + z*z/(2*n)) / denom
    half = (z*((p*(1-p) + z*z/(4*n))/n)**0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def estimate_count_random(F, trials=5000, rng=None):
    rng = rng or random.Random()
    V = list(_vars_set(F))
    if not V:
        sat = _eval_under(F, {})
        return {
            "n_vars": 0, "trials": trials, "k_sat": trials if sat else 0,
            "p_hat": 1.0 if sat else 0.0, "p_ci": (1.0,1.0) if sat else (0.0,0.0),
            "est_count": 1.0 if sat else 0.0, "est_ci": (1.0,1.0) if sat else (0.0,0.0),
            "method": "random_sampling"
        }
    k = 0
    for _ in range(trials):
        A = {v: bool(rng.getrandbits(1)) for v in V}
        if _eval_under(F, A):
            k += 1
    p_hat = k / trials
    lo, hi = _wilson_interval(k, trials)
    two_n = 2 ** len(V)
    return {
        "n_vars": len(V), "trials": trials, "k_sat": k,
        "p_hat": p_hat, "p_ci": (lo, hi),
        "est_count": p_hat * two_n, "est_ci": (lo * two_n, hi * two_n),
        "method": "random_sampling"
    }

def estimate_count_samplesat(F, runs=200, max_tries=30, max_flips=20000, noise=0.5, rng=None):
    rng = rng or random.Random()
    V = list(_vars_set(F))
    if not V:
        sat = _eval_under(F, {})
        return {
            "n_vars": 0, "runs": runs, "hits": runs if sat else 0,
            "success_rate": 1.0 if sat else 0.0, "est_count_naive": 1.0 if sat else 0.0,
            "method": "samplesat_runs"
        }
    hits = 0
    flips = []
    for _ in range(runs):
        res = sample_sat(F, max_tries=max_tries, max_flips=max_flips, noise=noise, rng=rng)
        if res.sat:
            hits += 1
            flips.append(res.flips_used)
    sr = hits / runs
    two_n = 2 ** len(V)
    return {
        "n_vars": len(V), "runs": runs, "hits": hits,
        "success_rate": sr, "avg_flips_if_hit": (sum(flips)/len(flips)) if flips else None,
        "est_count_naive": sr * two_n, "method": "samplesat_runs"
    }

# --------- Comparative Table: random vs SampleSAT ---------
def compare_estimators(formulas,  # list of (name, sympy_expr)
                       trials_random=5000,
                       runs_samplesat=200,
                       noise=0.5,
                       rng=None):
    """
    Build a comparison over multiple formulas.
    Returns a list of dict rows with:
      - name, n_vars
      - random: p_hat, p_ci, est_count, est_ci, trials
      - samplesat: success_rate, est_count_naive, runs, avg_flips_if_hit
    """
    rng = rng or random.Random()
    rows = []
    for name, F in formulas:
        r = estimate_count_random(F, trials=trials_random, rng=rng)
        s = estimate_count_samplesat(F, runs=runs_samplesat, noise=noise, rng=rng)
        rows.append({
            "name": name,
            "n_vars": r["n_vars"],
            "random_trials": r["trials"],
            "p_hat": r["p_hat"],
            "p_ci_lo": r["p_ci"][0],
            "p_ci_hi": r["p_ci"][1],
            "est_count_random": r["est_count"],
            "est_ci_lo": r["est_ci"][0],
            "est_ci_hi": r["est_ci"][1],
            "samplesat_runs": s["runs"],
            "success_rate": s["success_rate"],
            "avg_flips_if_hit": s["avg_flips_if_hit"],
            "est_count_samplesat": s["est_count_naive"]
        })
    return rows

def compare_estimators_df(formulas, trials_random=5000, runs_samplesat=200, noise=0.5, rng=None):
    try:
        import pandas as pd
    except Exception:
        raise RuntimeError("pandas non disponibile: usa compare_estimators(...) per avere una lista di dict.")
    rows = compare_estimators(formulas, trials_random, runs_samplesat, noise, rng)
    cols = ["name","n_vars",
            "random_trials","p_hat","p_ci_lo","p_ci_hi",
            "est_count_random","est_ci_lo","est_ci_hi",
            "samplesat_runs","success_rate","avg_flips_if_hit","est_count_samplesat"]
    return pd.DataFrame(rows)[cols]
