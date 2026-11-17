#!/usr/bin/env python3
"""Do people read transcripts as the models do? Replication of two rating validity studies.

    python3 replicate.py                 runs every analysis on data/, prints the tables, writes results/*.json
    python3 replicate.py --check         also compares the results with expected.json (the numbers of the report)
    python3 replicate.py --boot 200      fewer resamples (faster; intervals then differ from the report)

Needs Python 3.9 or later with numpy and scipy. Takes a few minutes. Every random draw is seeded, so two runs give
the same numbers.

The data (data/README.md) hold people's ratings of 250-word excerpts of machine transcripts and two language
models' scores of the same excerpts. They hold no text and nothing that identifies a rater. Results describe how
people and models rate the wording of excerpts; they are not facts about any person.

Contents
  1  reading the data
  2  estimators of the first study (one rater per excerpt and question, five raters of a closed set)
  3  estimators of the second study (every rater another mix of excerpts; rater levels taken out)
  4  first study: results fixed in advance, and analyses added afterwards
  5  second study: results fixed in advance
  6  second study: analyses added afterwards, first round
  7  second study: analyses added afterwards, second round
  8  a third model, d1.2-ultra, on both studies
  9  printing, writing, checking"""
import collections, csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent; DATA = HERE / "data"; K = 5; MODELS = ("jev14", "drex"); KINDS = ("video", "stream"); PERIODS = ("2015-2018", "2019-2021", "2022-2025")
ARMS = ("own answers only", "sample answers")            # arm 0 and arm 1; the arm of a set is a column of study2_sets.csv
LABELS = dict(F1="Direct, organized argument", F2="Positive tone versus disappointment", F3="Self-disclosure and self-criticism", F4="Care and regard for others", F5="Narrative versus rebuttal",
              F6="Energetic and angry versus calm", F7="Humor and imaginative language", F8="Hedging and open-mindedness versus certainty")
P5 = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
sb = lambda r1, k=K: k * r1 / (1 + (k - 1) * r1)          # Spearman-Brown: reliability of the mean of k raters from that of one


def verdict(rel, r):
    """The rule fixed in advance. rel: reliability of the mean of five raters; r: the better model's correlation with it."""
    return "usable" if rel >= 0.6 and r >= 0.5 else "unclear rubric" if rel < 0.4 else "misread" if rel >= 0.6 and r < 0.3 else "borderline"


def settled(point, rel, r): return {verdict(a, b) for a in rel for b in r} == {point}      # the whole interval on one side of every line


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0: return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def wcov(X, w): w = w / w.sum(); mu = (X * w[:, None]).sum(0); Xc = X - mu; return (Xc * w[:, None]).T @ Xc, mu


# ================================================================ 1  reading the data
def read_csv(name): return list(csv.DictReader(open(DATA / name, newline="", encoding="utf-8")))


def load():
    qrows = read_csv("questions.csv"); Q = [dict(id=r["question"], name=r["name"], dimension=r["dimension"], sign=int(r["sign"]), form=r["form_study2"], place=int(r["place_on_form_study2"]), rated_before=r["new_in_study2"] == "0",
                                               form1=r["form_study1"], place1=int(r["place_in_study1"]) if r["place_in_study1"] else None) for r in qrows]
    qids = [q["id"] for q in Q]; prow = read_csv("pool.csv"); fids = [c[5:] for c in prow[0] if c.startswith("site_")]; col = lambda pre, names: np.array([[float(r[pre + x]) for x in names] for r in prow])
    P = dict(cand=[r["candidate"] for r in prow], rid=np.array([r["recording"] for r in prow]), kind=np.array([r["kind"] for r in prow]), period=np.array([r["period"] for r in prow]), slot=np.array([int(r["passage"]) - 1 for r in prow]),
             w=np.array([float(r["weight"]) for r in prow]), fids=fids, QJ=col("jev_", qids), QD=col("drex_", qids), FJ=col("jev_", fids), FD=col("drex_", fids), site=col("site_", fids), QL=col("d12u_", qids), FL=col("d12u_", fids))
    first, second = {}, {}
    for i, (r, s) in enumerate(zip(P["rid"], P["slot"])):
        if s == 0: first[r] = i
        elif s == 1: second[r] = i
    P["pairs"] = np.array([(first[r], second[r]) for r in first if r in second]); pos = {c: i for i, c in enumerate(P["cand"])}

    # ---- first study
    e1 = read_csv("study1_excerpts.csv"); meta = {r["excerpt"]: dict(candidate=r["candidate"], part=r["part"], kind=r["kind"], period=r["period"], year=float(r["year"])) for r in e1}; Q1 = [q["id"] for q in sorted((q for q in Q if q["place1"]), key=lambda q: q["place1"])]
    rows1 = read_csv("study1_ratings.csv"); ex1 = sorted(meta); idx1 = np.array([pos[meta[e]["candidate"]] for e in ex1]); qc = {q: j for j, q in enumerate(qids)}; w = P["w"] / P["w"].sum(); pop = {}
    for q in Q1:
        a, b = P["QJ"][:, qc[q]], P["QD"][:, qc[q]]; ma, mb = (a * w).sum(), (b * w).sum(); sa, sd_ = math.sqrt((((a - ma) ** 2) * w).sum()), math.sqrt((((b - mb) ** 2) * w).sum())
        pop[q] = dict(mean_jev14=float(ma), sd_jev14=sa, mean_drex=float(mb), sd_drex=sd_, r_models=float(((a - ma) * (b - mb) * w).sum() / (sa * sd_)))
    by = {q: collections.defaultdict(list) for q in Q1}; who = collections.defaultdict(dict)
    for r in rows1:
        if r["rating"] != "": by[r["question"]][r["excerpt"]].append(float(r["rating"])); who[(r["excerpt"], r["question"])][r["rater"]] = float(r["rating"])
    S1 = dict(Q=Q1, form={q["id"]: q["form1"] for q in Q if q["place1"]}, dimension={q["id"]: q["dimension"] for q in Q}, ex=ex1, meta=meta, idx=idx1, rows=[r for r in rows1 if r["rating"] != ""], answers=len(rows1), raters=len({r["rater"] for r in rows1}), population=pop,
              key={e: {"jev14": {q: float(P["QJ"][i, qc[q]]) for q in Q1}, "drex": {q: float(P["QD"][i, qc[q]]) for q in Q1}} for e, i in zip(ex1, idx1)}, groups={q: [np.array(by[q][e]) for e in ex1] for q in Q1}, by_rater=who,
              s12={"jev14": P["QJ"][:, [qc[q] for q in Q1]], "drex": P["QD"][:, [qc[q] for q in Q1]]}, kind=np.array([meta[e]["kind"] for e in ex1]), year=np.array([meta[e]["year"] for e in ex1], float), random=np.array([meta[e]["part"] == "random" for e in ex1]))

    # ---- second study
    ex = [dict(id=r["excerpt"], candidate=r["candidate"], passage=int(r["passage"]), kind=r["kind"], period=r["period"], year=int(r["year"])) for r in read_csv("study2_excerpts.csv")]; idx = np.array([pos[e["candidate"]] for e in ex]); byrec = collections.defaultdict(list)
    for a, (e, i) in enumerate(zip(ex, idx)): e["rid"] = P["rid"][i]; byrec[e["rid"]].append(a)
    pairs = np.array(list(byrec.values())); mate = np.empty(len(ex), int); mate[pairs[:, 0]] = pairs[:, 1]; mate[pairs[:, 1]] = pairs[:, 0]; keys = [ex[p[0]]["kind"] for p in pairs]
    sets = {x: [[] for _ in range(100)] for x in "AB"}; info = {}       # a set's label is random; its round (a random label too), arm and first question are columns
    for r in read_csv("study2_sets.csv"): sets[r["form"]][int(r["set"])].append(r["excerpt"]); info[(r["form"], int(r["set"]))] = (int(r["round"]) - 1, int(r["arm"] == "sample answers"), int(r["first_question"]) - 1)
    F = dict(Q=Q, P=P, POP=population(P), ex=ex, ids={e["id"]: n for n, e in enumerate(ex)}, idx=idx, other=idx[mate], pairs=pairs, strata=[np.flatnonzero([k == s for k in keys]) for s in sorted(set(keys))], kind=np.array([e["kind"] for e in ex]),
             period=np.array([e["period"] for e in ex]), sets=sets, forms={x: [q["id"] for q in sorted((q for q in Q if q["form"] == x), key=lambda q: q["place"])] for x in "AB"}, fids=fids)
    rat = sorted(read_csv("study2_raters.csv"), key=lambda r: r["rater"]); num = {r["rater"]: n for n, r in enumerate(rat)}; rows = read_csv("study2_ratings.csv")
    D = dict(n_valid=len(rat), form=np.array([r["form"] for r in rat]), set=np.array([int(r["set"]) for r in rat], int), rater=np.array([num[r["rater"]] for r in rows], int),
             item=np.array([F["ids"][r["excerpt"]] for r in rows], int), qid=np.array([r["question"] for r in rows]), y=np.array([float(r["rating"]) if r["rating"] != "" else np.nan for r in rows]), position=np.array([int(r["position"]) for r in rows], int))
    D["round"], D["arm"], D["shift"] = (np.array([info[(f, s)][k] for f, s in zip(D["form"], D["set"])], int) for k in range(3)); per = collections.defaultdict(set)
    # time on the page: no rater's own time is released, only counts in steps of five minutes and summaries
    D["time"] = dict(summary={r["raters_of"]: {k: float(v) for k, v in r.items() if k != "raters_of"} for r in read_csv("study2_time_summary.csv")}, histogram=collections.defaultdict(list))
    for r in read_csv("study2_time_histogram.csv"): D["time"]["histogram"][r["arm"]].append((float(r["minutes_from"]), float(r["minutes_to"]), int(r["raters"])))
    for r, i in zip(D["rater"], D["item"]): per[int(r)].add(int(i))
    cnt = collections.Counter(zip(D["form"], D["set"])); bad = sum(per[k] != {F["ids"][i] for i in F["sets"][D["form"][k]][D["set"][k]]} for k in range(len(rat)))
    D["sessions"] = dict(valid=len(rat), raters_whose_excerpts_are_not_their_set=int(bad), sets_with_a_valid_rater=len(cnt), sets_with_two_or_more=sum(v > 1 for v in cnt.values()), sets_without=200 - len(cnt), valid_by_form={x: int((D["form"] == x).sum()) for x in "AB"},
                         valid_by_arm={ARMS[a]: int((D["arm"] == a).sum()) for a in (1, 0)})
    calls = read_csv("study2_model_calls.csv")
    return Q, P, S1, F, D, calls


# ================================================================ 2  estimators of the first study
def one_way(groups):
    """Reliability of one rating from a one-way analysis of variance (raters differ by excerpt), and the within-excerpt mean square."""
    k = np.mean([len(g) for g in groups]); n = len(groups); allv = np.concatenate(groups); gm = allv.mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (n - 1); msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (len(allv) - n)
    return float((msb - msw) / (msb + (k - 1) * msw)), float(msw)


def figures1(groups, XA, XB, random, pop):
    """All numbers of one question of the first study. groups: the ratings of each excerpt; XA, XB: the two models' scores;
    random: which excerpts are the random part; pop: means, spreads and correlation of the two scores in the population.
    A  the random part alone, directly.  B  all excerpts, rebuilt for the population from the regression of the mean
    rating on the two scores (the excerpts chosen by their scores leave that regression untouched)."""
    hm = np.array([g.mean() for g in groups]); n_r = np.array([len(g) for g in groups]); out = {}
    icc, _ = one_way([g for g, r in zip(groups, random) if r]); out["random_reliability_one"] = icc; out["random_reliability_mean"] = sb(icc)
    out["random_r_jev14"] = float(stats.spearmanr(XA[random], hm[random])[0]); out["random_r_drex"] = float(stats.spearmanr(XB[random], hm[random])[0])
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (n_r.sum() - len(groups)); k_h = len(groups) / (1 / n_r).sum()
    X = np.column_stack([np.ones(len(hm)), XA, XB]); beta = np.linalg.lstsq(X, hm, rcond=None)[0]; res = hm - X @ beta; s2 = res @ res / (len(hm) - 3); var_e = max(s2 - msw / k_h, 0.0)
    S = np.array([[pop["sd_jev14"] ** 2, pop["r_models"] * pop["sd_jev14"] * pop["sd_drex"]], [pop["r_models"] * pop["sd_jev14"] * pop["sd_drex"], pop["sd_drex"] ** 2]]); b = beta[1:]
    var_t = float(b @ S @ b + var_e); rel = var_t / (var_t + msw / K) if var_t > 0 else 0.0; rho = (S @ b) / np.sqrt(max(var_t, 1e-12) * np.diag(S)) * np.sqrt(rel)
    out.update(rebuilt_reliability_one=var_t / (var_t + msw) if var_t > 0 else 0.0, rebuilt_reliability_mean=rel, rebuilt_r_jev14=float(rho[0]), rebuilt_r_drex=float(rho[1]), rebuilt_difference=float(rho[0] - rho[1]))
    zA, zB = (XA - pop["mean_jev14"]) / pop["sd_jev14"], (XB - pop["mean_drex"]) / pop["sd_drex"]; G = np.column_stack([np.ones(len(hm)), (zA + zB) / 2, zA - zB]); g = np.linalg.lstsq(G, hm, rcond=None)[0]
    out["b_mean_of_models"] = float(g[1]); out["b_difference"] = float(g[2])                 # which model people side with where the two differ
    out["model_minus_people_jev14"] = float((XA - hm)[random].mean()); out["model_minus_people_drex"] = float((XB - hm)[random].mean())
    return out


def within(groups):
    """Noise variance of one rating (pooled within excerpts) and the mean of 1 / (number of ratings)."""
    n = np.array([len(g) for g in groups]); return float(sum(((g - g.mean()) ** 2).sum() for g in groups) / (n.sum() - len(groups))), float((1 / n).mean())


def rebuild(hm, msw, inv_n, X, Sigma, mu, combos=None):
    """Population figures from a sample chosen on the predictors X (first study; Section 'The estimator generalised').
    rating = T + noise, T = a + b'X + e, e unrelated to X. Then var(T) = b' Sigma b + var(e), cov(c'X, T) = c' Sigma b."""
    n, p = X.shape; xm = X.mean(0); Xc = X - xm; G = np.linalg.pinv(Xc.T @ Xc); yc = hm - hm.mean(); b = G @ (Xc.T @ yc); res = yc - Xc @ b; s2 = float(res @ res / max(n - p - 1, 1))
    var_e = max(s2 - msw * inv_n, 0.0); signal = max(float(b @ Sigma @ b) - s2 * float(np.trace(Sigma @ G)), 1e-12); var_t = signal + var_e
    rel1 = var_t / (var_t + msw); relK = var_t / (var_t + msw / K); mean_t = float(hm.mean() + b @ (mu - xm)); Sb = Sigma @ b
    out = dict(var_true=var_t, noise=msw, reliability_one=rel1, reliability_mean=relK, one_rater_with_panel=math.sqrt(rel1), people_mean=mean_t)
    for name, c in (combos or {}).items():
        c = np.asarray(c, float); vc = float(c @ Sigma @ c); rho = min(float(c @ Sb / math.sqrt(vc * var_t)), 1.0) if vc > 0 else float("nan"); r2 = min(rho * rho, 0.999999)
        out[name] = dict(with_panel=rho, with_mean=rho * math.sqrt(relK), raters_worth=(r2 / (1 - r2)) * msw / var_t, slope=float(c @ Sb / vc), intercept=float(mean_t - (c @ Sb / vc) * (c @ mu)), minus_people=float(c @ mu - mean_t))
    return out


# ================================================================ 3  estimators of the second study
def fit_crossed(y, rater, item, n_item, W=None):
    """Ratings of one question on one form: rating = excerpt mean + rater's level + rest, the excerpt means fixed, the
    levels random. Restricted maximum likelihood for the two variances, generalised least squares for the means.
    Returns the excerpt means with levels taken out, the noise variance of each, the variances of the levels and of the
    rest, and the coefficients of further fixed columns W (the arm). An excerpt with fewer than two ratings has no mean."""
    total = n_item; good = np.bincount(item, minlength=total) >= 2; keep = good[item]; y, item = y[keep], (np.cumsum(good) - 1)[item[keep]]; rater = np.unique(rater[keep], return_inverse=True)[1]; W = None if W is None else W[keep]; n_item = int(good.sum())
    N = len(y); nr = int(rater.max()) + 1; p = n_item + (0 if W is None else W.shape[1]); X = np.zeros((N, p)); X[np.arange(N), item] = 1
    if W is not None: X[:, n_item:] = W
    R = np.zeros((N, nr)); R[np.arange(N), rater] = 1; n_r = R.sum(0); RX = R.T @ X; Ry = R.T @ y; XX = X.T @ X; Xy = X.T @ y; yy = float(y @ y)
    def parts(loglam):
        lam = math.exp(loglam); c = lam / (1 + n_r * lam); A = XX - RX.T @ (c[:, None] * RX); b = Xy - RX.T @ (c * Ry); beta = np.linalg.solve(A, b); q = yy - float((c * Ry) @ Ry) - float(b @ beta)
        s2d = q / (N - p); return (N - p) * math.log(s2d) + float(np.log1p(n_r * lam).sum()) + float(np.linalg.slogdet(A)[1]), beta, s2d, A, lam
    best = minimize_scalar(lambda t: parts(t)[0], bounds=(-9.0, 4.0), method="bounded", options=dict(xatol=1e-4)); _, beta, s2d, A, lam = parts(best.x); C = s2d * np.linalg.inv(A)
    mean = np.full(total, np.nan); noise = mean.copy(); mean[good] = beta[:n_item]; noise[good] = np.diag(C)[:n_item]
    return dict(mean=mean, noise=noise, var_level=lam * s2d, var_rest=s2d, fixed=beta[n_item:], fixed_se=np.sqrt(np.diag(C)[n_item:]), excerpts_left_out=int(total - n_item))


def fits_of(D, F, who=None, with_arm=True, drop=None, only=None):
    """fit_crossed for the questions (all sixteen, or those of `only`). who: a mask over raters; drop: one rater left out.
    The arm is coded +0.5 (sample answers) and -0.5 (own answers only), so that an excerpt's mean stands for the middle."""
    out = {}; n = len(F["ex"])
    for q in (only or F["Q"]):
        m = (D["qid"] == q["id"]) & np.isfinite(D["y"])
        if who is not None: m &= who[D["rater"]]
        if drop is not None: m &= D["rater"] != drop
        W = (D["arm"][D["rater"][m]] - 0.5)[:, None] if with_arm else None
        out[q["id"]] = fit_crossed(D["y"][m], D["rater"][m], D["item"][m], n, W); out[q["id"]]["ratings_per_excerpt"] = np.bincount(D["item"][m], minlength=n)
    return out


def composites(fits, Q, fids):
    """Per dimension: the mean of its two questions' excerpt means, signs as the loadings, and its noise variance."""
    out = {}
    for f in fids:
        qs = [q for q in Q if q["dimension"] == f]; out[f] = (sum(q["sign"] * fits[q["id"]]["mean"] for q in qs) / len(qs), sum(fits[q["id"]]["noise"] for q in qs) / len(qs) ** 2)
    return out


def plain(h, c, c5, X):
    """Plain figures for excerpt means h with noise c: reliability of the mean of five raters and each model's
    correlation with that mean. c5: the noise a mean of five raters would have."""
    ok = np.isfinite(h); h, c, X = h[ok], c[ok], X[ok]; cbar = float(np.mean(c)); dv = max(float(np.var(h, ddof=1)) - cbar, 1e-12); att = math.sqrt((dv + cbar) / (dv + c5))
    return dict(reliability=dv / (dv + c5), var_true=dv, excerpts=int(ok.sum()), **{m: min(float(np.corrcoef(X[:, j], h)[0, 1]) * att, 1.0) for j, m in enumerate(MODELS)})


def five(fit):
    """Noise of a mean of five raters from a fit in which the excerpts have other numbers of ratings."""
    ok = np.isfinite(fit["mean"]); k = fit["ratings_per_excerpt"][ok]; return float(np.mean(fit["noise"][ok])) / (5 * float(np.mean(1 / k)))


PERM = [3, 4, 2, 0, 1]      # the predictors of a passage seen from its partner: own and partner's scores change places


def dimension(h, c, Z, Sigma, mu, pairs, k_ratio=1.0):
    """Figures of one dimension. h, c: composite per excerpt and its noise. Z: per excerpt both models' dimension scores of
    the passage, the site's score of the recording, both models' scores of the other passage; Sigma, mu: their covariance
    and means in the population; pairs: rows of the two passages of each recording. 'direct' holds the plain figures, on
    which the verdict rests; the rest is the estimate with the model's help. Stated for five raters per question."""
    n, p = Z.shape; zm = Z.mean(0); Zc = Z - zm; G = np.linalg.pinv(Zc.T @ Zc); yc = h - h.mean(); b = G @ (Zc.T @ yc); res = yc - Zc @ b; s2 = float(res @ res / (n - p - 1)); cbar = float(np.mean(c)); c5 = cbar * k_ratio
    var_u = max(s2 - cbar, 0.0); expl = max(float(b @ Sigma @ b) - s2 * float(np.trace(Sigma @ G)), 1e-12); var_t = expl + var_u; rel = var_t / (var_t + c5); Sb = Sigma @ b
    out = dict(var_true=var_t, noise_of_mean=c5, reliability=rel, people_mean=float(h.mean() + b @ (mu - zm)))
    for j, m in enumerate(MODELS):
        rho = min(float(Sb[j] / math.sqrt(Sigma[j, j] * var_t)), 1.0); out[m] = dict(with_panel=rho, with_mean=rho * math.sqrt(rel))
    out["verdict_with_model"] = verdict(rel, max(out[m]["with_mean"] for m in MODELS))
    SB = Sigma[:, PERM]; SB = (SB + SB.T) / 2; eta = float(np.mean(res[pairs[:, 0]] * res[pairs[:, 1]])) * n / (n - p - 1)
    var_p = min(max(float(b @ SB @ b) - s2 * float(np.trace(SB @ G)) + eta, 1e-12), var_t)
    out["recordings"] = dict(share_between_recordings=var_p / var_t, site_score_with_panel=min(float(Sb[2] / math.sqrt(Sigma[2, 2] * var_p)), 1.0), reliability_of_two_passages=var_p / (var_p + (var_t - var_p) / 2 + cbar / 2))
    dv = max(float(np.var(h, ddof=1)) - cbar, 1e-12); cp = float(np.cov(h[pairs[:, 0]], h[pairs[:, 1]])[0, 1]); hg = (h[pairs[:, 0]] + h[pairs[:, 1]]) / 2; att = math.sqrt((dv + cbar) / (dv + c5))
    out["direct"] = dict(reliability=dv / (dv + c5), share_between_recordings=min(max(cp, 0.0) / dv, 1.0), site_score_with_mean_of_recording=float(np.corrcoef(Z[pairs[:, 0], 2], hg)[0, 1]),
                         **{m: dict(with_mean=min(float(np.corrcoef(Z[:, j], h)[0, 1]) * att, 1.0)) for j, m in enumerate(MODELS)})
    out["verdict"] = verdict(out["direct"]["reliability"], max(out["direct"][m]["with_mean"] for m in MODELS))
    return out


def population(P):
    """Per dimension: the predictors of a passage and their covariance and means in the population: every recording of
    the pool with two passages once, each passage in turn, the two kinds in equal numbers."""
    pairs = P["pairs"]; pk = P["kind"][pairs[:, 0]]; wp = np.zeros(len(pairs)); out = []
    for k in KINDS: wp[pk == k] = 0.5 / (pk == k).sum()
    own = np.concatenate([pairs[:, 0], pairs[:, 1]]); other = np.concatenate([pairs[:, 1], pairs[:, 0]]); ww = np.concatenate([wp, wp]) / 2
    for f in range(len(P["fids"])):
        Z = lambda o, q, f=f: np.column_stack([P["FJ"][o, f], P["FD"][o, f], P["site"][o, f], P["FJ"][q, f], P["FD"][q, f]]); Sigma, mu = wcov(Z(own, other), ww); out.append(dict(Z=Z, Sigma=Sigma, mu=mu))
    return out


def figures2(comp, POP, idx, other, pairs, fids, kr):
    out = {}
    for k, f in enumerate(fids):
        h, c = comp[f]; ok = np.isfinite(h); pr = pairs if ok.all() else (np.cumsum(ok) - 1)[pairs[ok[pairs].all(1)]]
        out[f] = dimension(h[ok], c[ok], POP[k]["Z"](idx[ok], other[ok]), POP[k]["Sigma"], POP[k]["mu"], pr, kr); out[f]["excerpts"] = int(ok.sum()); out[f]["recordings_with_both_passages"] = len(pr)
    return out


FLAT = lambda r: dict(reliability=r["reliability"], jev14_with_panel=r["jev14"]["with_panel"], drex_with_panel=r["drex"]["with_panel"], jev14_with_mean=r["jev14"]["with_mean"], var_true=r["var_true"],
                      share_between_recordings=r["recordings"]["share_between_recordings"], site_score_with_panel=r["recordings"]["site_score_with_panel"],
                      direct_reliability=r["direct"]["reliability"], direct_with_mean=r["direct"]["jev14"]["with_mean"], direct_with_mean_drex=r["direct"]["drex"]["with_mean"], direct_share=r["direct"]["share_between_recordings"])


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); a, b = a[np.isfinite(a)], b[np.isfinite(b)]; va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b); d = float(a.mean() - b.mean()); se = math.sqrt(va + vb)
    t = float(stats.t.ppf(0.975, (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)))); return dict(first=float(a.mean()), second=float(b.mean()), difference=d, interval=[d - t * se, d + t * se], n=[len(a), len(b)])


def welch_summary(a, b):
    """Welch's interval from each group's mean, standard deviation and size."""
    va, vb = a["sd"] ** 2 / a["raters"], b["sd"] ** 2 / b["raters"]; d = a["mean"] - b["mean"]; se = math.sqrt(va + vb); t = float(stats.t.ppf(0.975, (va + vb) ** 2 / (va ** 2 / (a["raters"] - 1) + vb ** 2 / (b["raters"] - 1))))
    return dict(first=a["mean"], second=b["mean"], difference=d, interval=[d - t * se, d + t * se], n=[int(a["raters"]), int(b["raters"])])


def rater_table(D, F, fits):
    """Per valid rater: level, spread, 'cannot judge', agreement with the other raters of the same excerpts and questions,
    and how closely the ratings follow each model's question scores. Ratings and scores are centred within the question."""
    Q, P, idx = F["Q"], F["P"], F["idx"]; n, nr = len(F["ex"]), D["n_valid"]; col = {q["id"]: j for j, q in enumerate(Q)}; qj = np.array([col[q] for q in D["qid"]]); ok = np.isfinite(D["y"])
    zs = lambda X: (X - X.mean(0)) / X.std(0); MJ, MD = zs(P["QJ"][idx]), zs(P["QD"][idx]); S = np.zeros((n, len(Q))); N = np.zeros((n, len(Q))); np.add.at(S, (D["item"][ok], qj[ok]), D["y"][ok]); np.add.at(N, (D["item"][ok], qj[ok]), 1)
    gm = np.array([D["y"][ok & (qj == j)].mean() for j in range(len(Q))]); mean = np.column_stack([fits[q["id"]]["mean"] for q in Q])
    out = {k: np.full(nr, np.nan) for k in ("ratings", "cannot_judge", "spread", "level", "with_others", "with_jev14", "with_drex")}
    for r in range(nr):
        m = D["rater"] == r; mo = m & ok; i, j, y = D["item"][mo], qj[mo], D["y"][mo]; out["cannot_judge"][r] = float((m & ~ok).sum()) / max(int(m.sum()), 1); out["ratings"][r] = len(y)
        if len(y) < 8: continue
        yc = y - gm[j]; out["spread"][r] = float(y.std(ddof=1)); others = N[i, j] > 1
        if np.isfinite(mean[i, j]).any(): out["level"][r] = float(np.nanmean(y - mean[i, j]))
        if others.sum() >= 8 and yc[others].std() > 0: out["with_others"][r] = float(np.corrcoef(yc[others], (S[i, j][others] - y[others]) / (N[i, j][others] - 1) - gm[j][others])[0, 1])
        if yc.std() > 0: out["with_jev14"][r] = float(np.corrcoef(yc, MJ[i, j])[0, 1]); out["with_drex"][r] = float(np.corrcoef(yc, MD[i, j])[0, 1])
    return out


def kripp(units, metric):
    """Krippendorff's alpha. units: one list of ratings (0..4) per excerpt; units with fewer than two are left out."""
    o = np.zeros((5, 5))
    for u in units:
        if len(u) < 2: continue
        cnt = np.bincount(np.asarray(u, int), minlength=5).astype(float); m = len(u); o += (np.outer(cnt, cnt) - np.diag(cnt)) / (m - 1)
    nc = o.sum(1); n = nc.sum()
    if metric == "interval": d2 = np.array([[(c - k) ** 2 for k in range(5)] for c in range(5)], float)
    else: d2 = np.array([[(nc[min(c, k):max(c, k) + 1].sum() - (nc[c] + nc[k]) / 2) ** 2 for k in range(5)] for c in range(5)], float)
    return float(1 - (o * d2).sum() / ((np.outer(nc, nc) * d2).sum() / (n - 1)))


# ================================================================ 4  first study
def study1_fixed(S, B=1000, seed=1):
    """The results fixed in advance: per question the two sets of figures and the verdicts; then the two models over all
    twelve questions together. Intervals: excerpts resampled within part (random, chosen) and kind."""
    rng = np.random.default_rng(seed); meta = S["meta"]; out = {}
    for q in S["Q"]:
        by = collections.defaultdict(list)
        for r in S["rows"]:
            if r["question"] == q: by[r["excerpt"]].append(float(r["rating"]))
        ex = sorted(e for e in by if len(by[e]) >= 2); groups = [np.array(by[e]) for e in ex]; XA = np.array([S["key"][e]["jev14"][q] for e in ex]); XB = np.array([S["key"][e]["drex"][q] for e in ex]); random = np.array([meta[e]["part"] == "random" for e in ex]); pop = S["population"][q]
        est = figures1(groups, XA, XB, random, pop); strata = collections.defaultdict(list)
        for i, e in enumerate(ex): strata[(meta[e]["part"], meta[e]["kind"])].append(i)
        boot = collections.defaultdict(list)
        for _ in range(B):
            idx = np.concatenate([rng.choice(v, len(v)) for v in strata.values()]); f = figures1([groups[i] for i in idx], XA[idx], XB[idx], random[idx], pop)
            for k2, v in f.items(): boot[k2].append(v)
        ci = lambda k2: [float(np.percentile(boot[k2], 2.5)), float(np.percentile(boot[k2], 97.5))]
        res = dict(excerpts=len(ex), random_excerpts=int(random.sum()), raters_per_excerpt=float(np.mean([len(g) for g in groups])))
        res["random_part"] = dict(reliability_one_rater=est["random_reliability_one"], reliability_of_mean=est["random_reliability_mean"], reliability_of_mean_interval=ci("random_reliability_mean"),
                                  **{m: dict(rank_correlation=est["random_r_" + m], interval=ci("random_r_" + m), model_minus_people=est["model_minus_people_" + m]) for m in MODELS})
        res["rebuilt"] = dict(reliability_one_rater=est["rebuilt_reliability_one"], reliability_of_mean=est["rebuilt_reliability_mean"], reliability_of_mean_interval=ci("rebuilt_reliability_mean"),
                              **{m: dict(correlation=est["rebuilt_r_" + m], interval=ci("rebuilt_r_" + m)) for m in MODELS}, difference=est["rebuilt_difference"], difference_interval=ci("rebuilt_difference"))
        lo, hi = ci("b_difference"); res["which_model"] = dict(b_difference=est["b_difference"], interval=[lo, hi], people_side_with="jev14" if lo > 0 else "drex" if hi < 0 else "neither clearly")
        res["verdict"] = verdict(est["rebuilt_reliability_mean"], max(est["rebuilt_r_jev14"], est["rebuilt_r_drex"])); res["verdict_random_part_alone"] = verdict(est["random_reliability_mean"], max(est["random_r_jev14"], est["random_r_drex"]))
        out[q] = res
    # the two models over all questions at once: the same excerpts for every question in a draw
    rng = np.random.default_rng(seed); qs = S["Q"]; by = {q: collections.defaultdict(list) for q in qs}
    for r in S["rows"]: by[r["question"]][r["excerpt"]].append(float(r["rating"]))
    ex = sorted(e for e in meta if all(len(by[q][e]) >= 2 for q in qs)); random = np.array([meta[e]["part"] == "random" for e in ex]); strata = collections.defaultdict(list)
    for i, e in enumerate(ex): strata[(meta[e]["part"], meta[e]["kind"])].append(i)
    G = {q: [np.array(by[q][e]) for e in ex] for q in qs}; XA = {q: np.array([S["key"][e]["jev14"][q] for e in ex]) for q in qs}; XB = {q: np.array([S["key"][e]["drex"][q] for e in ex]) for q in qs}
    names = ("mean_correlation_jev14", "mean_correlation_drex", "mean_difference_of_correlations", "mean_b_difference", "questions_where_jev14_is_higher")
    def stat(idx):
        f = [figures1([G[q][i] for i in idx], XA[q][idx], XB[q][idx], random[idx], S["population"][q]) for q in qs]
        return [float(np.mean([x[k] for x in f])) for k in ("rebuilt_r_jev14", "rebuilt_r_drex", "rebuilt_difference", "b_difference")] + [float(sum(x["rebuilt_difference"] > 0 for x in f))]
    est = stat(np.arange(len(ex))); boot = np.array([stat(np.concatenate([rng.choice(v, len(v)) for v in strata.values()])) for _ in range(B)])
    tog = dict(excerpts=len(ex), questions=len(qs), **{n: est[i] for i, n in enumerate(names)}, **{n + "_interval": [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))] for i, n in enumerate(names)})
    return out, tog


def study1_further(S, P, B=1000, seed=20261115):
    """Added after the first study's results had been read: each model against the consensus of an unlimited panel and
    one rater against it; levels; the dimensions from their marker questions; known groups."""
    rng = np.random.default_rng(seed); ex = S["ex"]; meta = S["meta"]; st = collections.defaultdict(list); w = P["w"]; rnd = S["random"]; Q1 = S["Q"]; out = dict(questions={})
    for i, e in enumerate(ex): st[(meta[e]["part"], meta[e]["kind"])].append(i)
    strata = [np.array(v) for v in st.values()]; draws = [np.concatenate([rng.choice(v, len(v)) for v in strata]) for _ in range(B)]; kept, ests = {}, {}
    for j, q in enumerate(Q1):
        groups = S["groups"][q]; hm_all = np.array([g.mean() for g in groups]); XA, XB = S["s12"]["jev14"][S["idx"], j], S["s12"]["drex"][S["idx"], j]; X2 = np.column_stack([XA, XB]); pop = S["population"][q]
        Sig2, mu2 = wcov(np.column_stack([S["s12"]["jev14"][:, j], S["s12"]["drex"][:, j]]), w); combos = dict(jev14=[1, 0], drex=[0, 1])
        def one(idx):
            g = [groups[i] for i in idx]; msw, inv_n = within(g); f = figures1(g, XA[idx], XB[idx], rnd[idx], pop); main = rebuild(hm_all[idx], msw, inv_n, X2[idx], Sig2, mu2, combos)
            return dict(f=f, main=main, verdict=verdict(f["rebuilt_reliability_mean"], max(f["rebuilt_r_jev14"], f["rebuilt_r_drex"])))
        est = one(np.arange(len(ex))); boot = [one(i) for i in draws]; kept[q] = boot; ests[q] = est; ci = lambda get: P5([get(b) for b in boot])
        out["questions"][q] = dict(resampling=dict(verdicts={v: float(np.mean([b["verdict"] == v for b in boot])) for v in ("usable", "borderline", "unclear rubric", "misread")}),
            panel=dict(one_rater=est["main"]["one_rater_with_panel"], one_rater_interval=ci(lambda b: b["main"]["one_rater_with_panel"]),
                       **{m: dict(correlation=est["main"][m]["with_panel"], interval=ci(lambda b, m=m: b["main"][m]["with_panel"]), raters_worth=est["main"][m]["raters_worth"]) for m in MODELS}),
            levels=dict(people_mean=est["main"]["people_mean"], **{m: dict(model_mean=float(mu2[i]), minus_people=est["main"][m]["minus_people"], minus_people_interval=ci(lambda b, m=m: b["main"][m]["minus_people"]), slope=est["main"][m]["slope"]) for i, m in enumerate(MODELS)}))
    over = lambda get: dict(value=float(np.mean([get(ests[q]) for q in Q1])), interval=P5([np.mean([get(kept[q][t]) for q in Q1]) for t in range(B)]))
    out["together"] = dict(with_panel={m: over(lambda e, m=m: e["main"][m]["with_panel"]) for m in MODELS}, with_panel_one_rater=over(lambda e: e["main"]["one_rater_with_panel"]),
                           questions_where_model_beats_one_rater={m: int(sum(ests[q]["main"][m]["with_panel"] > ests[q]["main"]["one_rater_with_panel"] for q in Q1)) for m in MODELS})
    # ---- the dimensions from their one or two marker questions (two markers: F1, F2, F3, F6)
    marks_of = collections.defaultdict(list)
    for q in sorted(Q1, key=lambda q: [x["id"] for x in S["allq"]].index(q)): marks_of[S["dimension"][q]].append((q, S["sign"][q]))
    out["dimensions"] = {}; fids = P["fids"]
    for f in fids:
        marks = marks_of[f]; fi = fids.index(f); qi = [Q1.index(q) for q, _ in marks]; signs = np.array([s for _, s in marks], float); nm = len(marks); same_form = nm == 2 and len({S["form"][q] for q, _ in marks}) == 1
        if same_form:                        # the same raters answer both questions: one composite per rater
            groups = []
            for e in ex: a, b = S["by_rater"][(e, marks[0][0])], S["by_rater"][(e, marks[1][0])]; groups.append(np.array([(signs[0] * a[p] + signs[1] * b[p]) / 2 for p in a if p in b]))
            comp = lambda idx: (np.array([groups[i].mean() for i in idx]), *within([groups[i] for i in idx]))
        else:                                # one question, or two answered by different raters: independent noise
            gs = [S["groups"][q] for q, _ in marks]
            def comp(idx, gs=gs, signs=signs, nm=nm):
                hm = sum(s * np.array([g[i].mean() for i in idx]) for s, g in zip(signs, gs)) / nm; parts = [within([g[i] for i in idx]) for g in gs]; return hm, sum(m for m, _ in parts) / nm ** 2, float(np.mean([v for _, v in parts]))
        Pm = np.column_stack([S["s12"]["jev14"][:, qi], S["s12"]["drex"][:, qi], P["FJ"][:, fi], P["FD"][:, fi]]); Sig, mu = wcov(Pm, w); X = Pm[S["idx"]]; p = Pm.shape[1]; unit = lambda i: np.eye(p)[i]; combos = dict(dimension_jev14=unit(p - 2), dimension_drex=unit(p - 1))
        def one(idx): hm, msw, inv_n = comp(idx); return rebuild(hm, msw, inv_n, X[idx], Sig, mu, combos)
        est = one(np.arange(len(ex))); boot = [one(i) for i in draws]; ci = lambda get: P5([get(b) for b in boot])
        out["dimensions"][f] = dict(markers=[q for q, _ in marks], names=[S["name"][q] for q, _ in marks], raters="the same five answer both" if same_form else "five per question", reliability_of_mean=est["reliability_mean"], reliability_of_mean_interval=ci(lambda b: b["reliability_mean"]),
            **{k: dict(with_mean=est[k]["with_mean"], with_mean_interval=ci(lambda b, k=k: b[k]["with_mean"]), with_panel=est[k]["with_panel"]) for k in combos},
            by_the_rule_for_questions=verdict(est["reliability_mean"], max(est["dimension_jev14"]["with_mean"], est["dimension_drex"]["with_mean"])),
            usable_in_resamples=float(np.mean([verdict(b["reliability_mean"], max(b["dimension_jev14"]["with_mean"], b["dimension_drex"]["with_mean"])) == "usable" for b in boot])))
    # ---- known groups: streams against videos, and the year, in units of each measure's own spread
    kind = (S["kind"] == "stream").astype(float); year = S["year"]; out["known_groups"] = {}
    for scope, mask in (("random_part", rnd), ("all_excerpts", np.ones(len(ex), bool))):
        ii = np.flatnonzero(mask); strata2 = [ii[kind[ii] == 0], ii[kind[ii] == 1]]
        def stats_of(idx, y):
            y = y[idx]; k = kind[idx]; sp = math.sqrt(((y[k == 0] - y[k == 0].mean()) ** 2).sum() + ((y[k == 1] - y[k == 1].mean()) ** 2).sum()) / math.sqrt(len(y) - 2); d = (y[k == 1].mean() - y[k == 0].mean()) / sp
            Z = np.column_stack([np.ones(len(y)), k, year[idx] - year[idx].mean()]); return d, np.linalg.lstsq(Z, y / sp, rcond=None)[0][2]
        table = {q: dict(people=np.array([g.mean() for g in S["groups"][q]]), jev14=S["s12"]["jev14"][S["idx"], j], drex=S["s12"]["drex"][S["idx"], j]) for j, q in enumerate(Q1)}
        draws2 = [np.concatenate([rng.choice(s, len(s)) for s in strata2]) for _ in range(max(B // 4, 50))]
        def across(idx):
            v = {who: np.array([stats_of(idx, table[q][who]) for q in Q1]) for who in ("people", "jev14", "drex")}
            return {"%s_%s" % (what, m): float(np.corrcoef(v["people"][:, c], v[m][:, c])[0, 1]) for c, what in ((0, "kind"), (1, "year")) for m in MODELS}
        est = across(ii); bs = [across(i) for i in draws2]
        out["known_groups"][scope] = dict(excerpts=len(ii), questions={q: {who: dict(stream_minus_video=float(stats_of(ii, y)[0]), per_year=float(stats_of(ii, y)[1])) for who, y in table[q].items()} for q in Q1},
                                          across_questions={k: dict(value=v, interval=P5([b[k] for b in bs])) for k, v in est.items()})
    return out


# ================================================================ 5  second study: results fixed in advance
def study2_fixed(F, D, first, B=1000, seed=20261122):
    """Per dimension the plain figures, the verdict, settled or at the line, the figures with the model's help and the
    recordings; then what was fixed in advance as secondary. Intervals: recordings resampled within their kind, both
    passages together. first: the first study's figures per question (for the comparison column)."""
    Q, P, POP, ex, idx, other, pairs, strata, fids = F["Q"], F["P"], F["POP"], F["ex"], F["idx"], F["other"], F["pairs"], F["strata"], F["fids"]; n = len(ex); med = lambda v: float(np.median(v)); pct = lambda v, p: float(np.percentile(v, p))
    kr = D["n_valid"] / sum(len(v) for v in F["sets"].values())           # raters per excerpt and form over five
    fits = fits_of(D, F); comp = composites(fits, Q, fids); res = figures2(comp, POP, idx, other, pairs, fids, kr); okr = np.isfinite(D["y"])
    out = dict(sessions=D["sessions"], raters_per_excerpt_and_form_over_five=kr, resamples=B)
    TS = D["time"]["summary"]; out["minutes_on_the_page"] = dict(valid=dict(median=TS["all"]["median"], quartiles=[TS["all"]["first_quartile"], TS["all"]["third_quartile"]], over_25_minutes=TS["all"]["share_over_25_minutes"]),
                                      by_arm={ARMS[a]: dict(raters=int(TS[ARMS[a]]["raters"]), median=TS[ARMS[a]]["median"]) for a in (1, 0)})
    rng = np.random.default_rng(seed); picks = []; draws = {f: collections.defaultdict(list) for f in fids}
    for _ in range(B):
        pick = np.concatenate([rng.choice(v, len(v)) for v in strata]); rows = pairs[pick].ravel(); pr = np.arange(len(rows)).reshape(-1, 2); picks.append(pick)
        rb = figures2({f: (comp[f][0][rows], comp[f][1][rows]) for f in fids}, POP, idx[rows], other[rows], pr, fids, kr)
        for f in fids:
            for k, v in FLAT(rb[f]).items(): draws[f][k].append(v)
            draws[f]["verdict"].append(rb[f]["verdict"])
    ci = {f: {k: P5(v) for k, v in draws[f].items() if k != "verdict"} for f in fids}; out["dimensions"] = {}
    for f in fids:
        r, c = res[f], ci[f]; qs = [q for q in Q if q["dimension"] == f]; rc = [max(c["direct_with_mean"][0], c["direct_with_mean_drex"][0]), max(c["direct_with_mean"][1], c["direct_with_mean_drex"][1])]
        st = settled(r["verdict"], c["direct_reliability"], rc); vc = collections.Counter(draws[f]["verdict"])
        out["dimensions"][f] = dict(label=LABELS[f], questions=[q["id"] for q in qs], excerpts=r["excerpts"], recordings_with_both_passages=r["recordings_with_both_passages"], people_mean=r["people_mean"],
            plain=dict(reliability=r["direct"]["reliability"], reliability_interval=c["direct_reliability"], jev14_with_mean=r["direct"]["jev14"]["with_mean"], jev14_interval=c["direct_with_mean"], drex_with_mean=r["direct"]["drex"]["with_mean"], drex_interval=c["direct_with_mean_drex"],
                       better_model_interval=rc, verdict=r["verdict"], settled=bool(st), verdict_in_resamples={v: n_ / B for v, n_ in vc.most_common()}),
            with_the_model=dict(reliability=r["reliability"], reliability_interval=c["reliability"], var_true=r["var_true"], noise_of_mean=r["noise_of_mean"], jev14_with_panel=r["jev14"]["with_panel"], jev14_with_panel_interval=c["jev14_with_panel"],
                                drex_with_panel=r["drex"]["with_panel"], drex_with_panel_interval=c["drex_with_panel"], jev14_with_mean=r["jev14"]["with_mean"], jev14_with_mean_interval=c["jev14_with_mean"], drex_with_mean=r["drex"]["with_mean"], verdict=r["verdict_with_model"]),
            recordings=dict(share_between_recordings_plain=r["direct"]["share_between_recordings"], share_plain_interval=c["direct_share"], share_between_recordings=r["recordings"]["share_between_recordings"], share_interval=c["share_between_recordings"],
                            site_score_with_panel=r["recordings"]["site_score_with_panel"], site_score_interval=c["site_score_with_panel"], site_score_with_mean_of_two_passages=r["direct"]["site_score_with_mean_of_recording"], reliability_of_two_passages=r["recordings"]["reliability_of_two_passages"]))
    usable = np.array([[v == "usable" for v in draws[f]["verdict"]] for f in fids]).sum(0)
    out["dimensions_usable"] = dict(plain=sum(res[f]["verdict"] == "usable" for f in fids), settled_as_usable=sum(d_["plain"]["settled"] and d_["plain"]["verdict"] == "usable" for d_ in out["dimensions"].values()), in_resamples=P5(usable), with_the_model=sum(res[f]["verdict_with_model"] == "usable" for f in fids))
    # ---- the two models
    dj = np.array([draws[f]["direct_with_mean"] for f in fids]); dd = np.array([draws[f]["direct_with_mean_drex"] for f in fids]); rl = np.array([draws[f]["direct_reliability"] for f in fids])
    pj, pd_ = np.array([res[f]["direct"]["jev14"]["with_mean"] for f in fids]), np.array([res[f]["direct"]["drex"]["with_mean"] for f in fids])
    out["models"] = dict(per_dimension={f: dict(difference_jev14_minus_drex=float(pj[k] - pd_[k]), interval=P5(dj[k] - dd[k])) for k, f in enumerate(fids)},
                         over_the_eight=dict(jev14=float(pj.mean()), jev14_interval=P5(dj.mean(0)), drex=float(pd_.mean()), drex_interval=P5(dd.mean(0)), difference=float((pj - pd_).mean()), difference_interval=P5((dj - dd).mean(0)),
                                             reliability=float(np.mean([res[f]["direct"]["reliability"] for f in fids])), reliability_interval=P5(rl.mean(0))))
    # ---- the sixteen questions one by one
    out["questions"] = {}
    for j, q in enumerate(Q):
        ft = fits[q["id"]]; X = np.column_stack([P["QJ"][idx, j], P["QD"][idx, j]]); c5 = float(np.nanmean(ft["noise"])) * kr; pl = plain(ft["mean"], ft["noise"], c5, X); m = D["qid"] == q["id"]; bt = collections.defaultdict(list)
        for pick in picks:
            rows = pairs[pick].ravel(); b = plain(ft["mean"][rows], ft["noise"][rows], float(np.nanmean(ft["noise"][rows])) * kr, X[rows])
            for kk in ("reliability", "jev14", "drex"): bt[kk].append(b[kk])
        old = first.get(q["id"]); o = old and dict(reliability=old["rebuilt"]["reliability_of_mean"], jev14=old["rebuilt"]["jev14"]["correlation"], drex=old["rebuilt"]["drex"]["correlation"], verdict=old["verdict"])
        out["questions"][q["id"]] = dict(name=q["name"], dimension=q["dimension"], form=q["form"], sign=q["sign"], ratings=int((m & okr).sum()), cannot_judge=int((m & ~okr).sum()), excerpts_left_out=ft["excerpts_left_out"], sd_levels=math.sqrt(ft["var_level"]), sd_rest=math.sqrt(ft["var_rest"]),
            reliability_of_one_rating=pl["var_true"] / (pl["var_true"] + ft["var_rest"]), arm_effect_sample_answers_minus_own=float(ft["fixed"][0]), arm_effect_se=float(ft["fixed_se"][0]),
            reliability=pl["reliability"], reliability_interval=P5(bt["reliability"]), jev14_with_mean=pl["jev14"], jev14_interval=P5(bt["jev14"]), drex_with_mean=pl["drex"], drex_interval=P5(bt["drex"]), verdict_not_used=verdict(pl["reliability"], max(pl["jev14"], pl["drex"])), first_study=o or None)
    cjr = np.array([((D["rater"] == r) & ~okr).sum() for r in range(D["n_valid"])]); out["cannot_judge"] = dict(share=float(1 - okr.mean()), answers=int((~okr).sum()), raters_who_used_it=int((cjr > 0).sum()), by_the_ten_who_used_it_most=int(np.sort(cjr)[-10:].sum()))
    # ---- the two arms
    RT = rater_table(D, F, fits); a1, a0 = D["arm"] == 1, D["arm"] == 0; out["arms"] = dict(order="first: sample answers; second: own answers only", raters={k: welch(v[a1], v[a0]) for k, v in RT.items() if k != "ratings"}, minutes=welch_summary(D["time"]["summary"][ARMS[1]], D["time"]["summary"][ARMS[0]]),
        level_per_question={q["id"]: dict(sample_answers_minus_own=float(fits[q["id"]]["fixed"][0]), se=float(fits[q["id"]]["fixed_se"][0])) for q in Q}, dimensions={})
    for a in (1, 0):
        fa = fits_of(D, F, D["arm"] == a, with_arm=False); ca = composites(fa, Q, fids); row = {}
        for k, f in enumerate(fids):
            qs = [q for q in Q if q["dimension"] == f]; c5 = sum(five(fa[q["id"]]) for q in qs) / len(qs) ** 2; X = np.column_stack([P["FJ"][idx, k], P["FD"][idx, k]]); h, c = ca[f]; pl = plain(h, c, c5, X); bt = collections.defaultdict(list)
            for pick in picks[:min(B, 500)]:
                rows = pairs[pick].ravel(); b = plain(h[rows], c[rows], c5, X[rows])
                for kk in ("reliability", "jev14", "drex"): bt[kk].append(b[kk])
            row[f] = dict(excerpts=pl["excerpts"], reliability=pl["reliability"], reliability_interval=P5(bt["reliability"]), jev14_with_mean=pl["jev14"], jev14_interval=P5(bt["jev14"]), drex_with_mean=pl["drex"], drex_interval=P5(bt["drex"]))
        out["arms"]["dimensions"][ARMS[a]] = row
    # ---- parts of the excerpts
    def part(mask):
        o = {}
        for k, f in enumerate(fids):
            h, c = comp[f]; X = np.column_stack([P["FJ"][idx, k], P["FD"][idx, k]]); m = mask & np.isfinite(h); pl = plain(h[m], c[m], float(np.mean(c[m])) * kr, X[m]); bt = collections.defaultdict(list)
            for pick in picks:
                rows = pairs[pick].ravel(); rows = rows[m[rows]]
                if len(rows) < 12: continue
                b = plain(h[rows], c[rows], float(np.mean(c[rows])) * kr, X[rows])
                for kk in ("reliability", "jev14", "drex"): bt[kk].append(b[kk])
            o[f] = dict(excerpts=pl["excerpts"], reliability=pl["reliability"], reliability_interval=P5(bt["reliability"]), jev14_with_mean=pl["jev14"], jev14_interval=P5(bt["jev14"]), drex_with_mean=pl["drex"], drex_interval=P5(bt["drex"]), verdict_not_used=verdict(pl["reliability"], max(pl["jev14"], pl["drex"])))
        return o
    out["kinds"] = {kd: part(F["kind"] == kd) for kd in KINDS}; out["periods"] = {p: part(F["period"] == p) for p in PERIODS}
    out["rater_levels"] = dict(mean_sd_of_levels=float(np.mean([math.sqrt(fits[q["id"]]["var_level"]) for q in Q])), mean_sd_of_the_rest=float(np.mean([math.sqrt(fits[q["id"]]["var_rest"]) for q in Q])))
    return out, dict(fits=fits, comp=comp, kr=kr, RT=RT)


def repeat_calls(calls, Q, fids):
    """Four calls of each model on each excerpt: spread between calls of the same text, and the reliability of one call."""
    out = {}
    for m, name in (("jev14", "jev-1.14.0"), ("drex", "drex-max"), ("d12u", "d1.2-ultra")):
        rows = [r for r in calls if r["model"] == name]; exs = sorted({r["excerpt"] for r in rows}); pos = {e: i for i, e in enumerate(exs)}
        def table(colname):
            V = np.full((len(exs), 4), np.nan)
            for r in rows: V[pos[r["excerpt"]], int(r["call"]) - 1] = float(r[colname]) if r[colname] != "" else np.nan
            V = V[np.isfinite(V).all(1)]; wi = float(np.sqrt(V.var(1, ddof=1).mean())); b2 = max(float(V.mean(1).var(ddof=1)) - wi ** 2 / 4, 0.0); return dict(within=wi, between=math.sqrt(b2), reliability_of_one_call=b2 / (b2 + wi ** 2))
        out[m] = dict(questions={q["id"]: table(q["id"]) for q in Q}, dimensions={f: table(f) for f in fids})
    return out


# ================================================================ 6  second study: added afterwards, first round
def study2_further(F, D, res2, ctx, first_dims, B=1000, seed=20261123):
    Q, P, ex, idx, pairs, strata, fids = F["Q"], F["P"], F["ex"], F["idx"], F["pairs"], F["strata"], F["fids"]; n = len(ex); kr, fits, comp, RT = ctx["kr"], ctx["fits"], ctx["comp"], ctx["RT"]
    rng = np.random.default_rng(seed); boots = [pairs[np.concatenate([rng.choice(v, len(v)) for v in strata])].ravel() for _ in range(B)]
    QJ, QD, FJ, FD = P["QJ"][idx], P["QD"][idx], P["FJ"][idx], P["FD"][idx]; MQ = dict(jev14=QJ, drex=QD); MF = dict(jev14=FJ, drex=FD); kind = F["kind"]; year = np.array([e["year"] for e in ex], float); stream = (kind == "stream").astype(float)
    okr = np.isfinite(D["y"]); cells = collections.Counter(zip(D["item"][okr].tolist(), D["qid"][okr].tolist())); per_cell = collections.Counter(cells.get((i, q["id"]), 0) for i in range(n) for q in Q)
    out = dict(valid_raters=int(D["n_valid"]), resamples=B, data=dict(answers=int(len(D["y"])), ratings=int(okr.sum()), cannot_judge=int((~okr).sum()), cells=n * len(Q), ratings_per_cell={str(k): int(v) for k, v in sorted(per_cell.items())}))
    # ---- sources of a rating
    out["sources"] = {}
    for q in Q:
        ft = fits[q["id"]]; ok = np.isfinite(ft["mean"]); cbar = float(np.mean(ft["noise"][ok])); dv = max(float(np.var(ft["mean"][ok], ddof=1)) - cbar, 1e-12); vl, vr = float(ft["var_level"]), float(ft["var_rest"]); tot = dv + vl + vr
        out["sources"][q["id"]] = dict(name=q["name"], dimension=q["dimension"], form=q["form"], var_excerpt=dv, var_level=vl, var_rest=vr, share_excerpt=dv / tot, share_level=vl / tot, share_rest=vr / tot,
                                       reliability_levels_out=dv / (dv + cbar * kr), reliability_levels_in=dv / (dv + (vl + vr) / 5))
    out["sources_dimensions"] = {}
    for f in fids:
        h, c = comp[f]; ok = np.isfinite(h); cbar = float(np.mean(c[ok])); dv = max(float(np.var(h[ok], ddof=1)) - cbar, 1e-12); qs = [q for q in Q if q["dimension"] == f]; nin = sum(fits[q["id"]]["var_level"] + fits[q["id"]]["var_rest"] for q in qs) / (5 * len(qs) ** 2)
        out["sources_dimensions"][f] = dict(reliability_levels_out=dv / (dv + cbar * kr), reliability_levels_in=dv / (dv + nin))
    # ---- do the two markers of a dimension go together
    out["markers"] = {}
    for f in fids:
        qa, qb = [q for q in Q if q["dimension"] == f]; ja, jb = Q.index(qa), Q.index(qb); ha, hb = qa["sign"] * fits[qa["id"]]["mean"], qb["sign"] * fits[qb["id"]]["mean"]
        def rel(ft, rows=None):
            m, c = ft["mean"], ft["noise"]; m, c = (m, c) if rows is None else (m[rows], c[rows]); ok = np.isfinite(m); cb = float(np.mean(c[ok])); v = float(np.var(m[ok], ddof=1)); return max(v - cb, 1e-12) / v
        r = corr(ha, hb); ra, rb = rel(fits[qa["id"]]), rel(fits[qb["id"]]); bt = [corr(ha[rows], hb[rows]) for rows in boots]; bd = [corr(ha[rows], hb[rows]) / math.sqrt(rel(fits[qa["id"]], rows) * rel(fits[qb["id"]], rows)) for rows in boots]
        o = dict(questions=[qa["id"], qb["id"]], names=[qa["name"], qb["name"]], people=r, people_interval=P5(bt), people_without_rater_noise=min(r / math.sqrt(ra * rb), 1.0), people_without_rater_noise_interval=[min(v, 1.0) for v in P5(bd)])
        for m in MODELS:
            x, y = qa["sign"] * MQ[m][:, ja], qb["sign"] * MQ[m][:, jb]; o[m] = corr(x, y); o[m + "_interval"] = P5([corr(x[rows], y[rows]) for rows in boots])
            XA, XB = qa["sign"] * (P["QJ"] if m == "jev14" else P["QD"])[:, ja], qb["sign"] * (P["QJ"] if m == "jev14" else P["QD"])[:, jb]; w = P["w"] / P["w"].sum(); ma, mb = (XA * w).sum(), (XB * w).sum()
            o[m + "_in_the_pool"] = float(((XA - ma) * (XB - mb) * w).sum() / math.sqrt((((XA - ma) ** 2) * w).sum() * (((XB - mb) ** 2) * w).sum()))
        out["markers"][f] = o
    # ---- every model's dimension score against people's composite of every dimension
    H = np.column_stack([comp[f][0] for f in fids]); nf = len(fids); out["cross"] = dict(order=list(fids))
    def cross(M, rows=None):
        Hh, Mm = (H, M) if rows is None else (H[rows], M[rows]); return np.array([[corr(Mm[:, j], Hh[:, k]) for k in range(nf)] for j in range(nf)])
    for m in MODELS:
        C = cross(MF[m]); off = C - np.diag(np.diag(C)); dg = np.diag(C); rowmax = np.abs(off).max(1); colmax = np.abs(off).max(0); bt = np.array([cross(MF[m], rows) for rows in boots[:min(B, 500)]])
        gap = np.array([np.diag(c) - np.maximum(np.abs(c - np.diag(np.diag(c))).max(1), np.abs(c - np.diag(np.diag(c))).max(0)) for c in bt])
        out["cross"][m] = dict(matrix=C.tolist(), own=dg.tolist(), largest_other_in_row=rowmax.tolist(), largest_other_in_column=colmax.tolist(), own_is_largest=[bool(dg[k] > max(rowmax[k], colmax[k])) for k in range(nf)],
                               own_minus_largest_other=(dg - np.maximum(rowmax, colmax)).tolist(), own_minus_largest_other_interval=[P5(gap[:, k]) for k in range(nf)], mean_own=float(dg.mean()), mean_other=float(np.abs(off).sum() / (nf * (nf - 1))))
    Cp = np.array([[corr(H[:, a], H[:, b]) for b in range(nf)] for a in range(nf)]); iu = np.triu_indices(nf, 1); out["cross"]["people"] = dict(matrix=Cp.tolist())
    for m in MODELS:
        Cm = np.corrcoef(MF[m].T); out["cross"][m]["among_its_own_dimensions"] = Cm.tolist(); out["cross"][m]["pattern_with_people"] = corr(Cm[iu], Cp[iu])
    # ---- levels
    out["levels"] = {}
    for j, q in enumerate(Q):
        h = fits[q["id"]]["mean"]; ok = np.isfinite(h); o = dict(name=q["name"], dimension=q["dimension"], people=float(h[ok].mean()), people_sd_of_excerpt_means=float(h[ok].std(ddof=1)))
        for m in MODELS:
            x = MQ[m][:, j]; d = float(x[ok].mean() - h[ok].mean()); sl = float(np.polyfit(x[ok], h[ok], 1)[0]); bd, bs = [], []
            for rows in boots:
                k2 = np.isfinite(h[rows]); xr, hr = x[rows][k2], h[rows][k2]; bd.append(float(xr.mean() - hr.mean())); bs.append(float(np.polyfit(xr, hr, 1)[0]))
            o[m] = dict(mean=float(x[ok].mean()), sd=float(x[ok].std(ddof=1)), model_minus_people=d, interval=P5(bd), slope_people_per_model_point=sl, slope_interval=P5(bs))
        out["levels"][q["id"]] = o
    for m in MODELS:
        d = np.array([out["levels"][q["id"]][m]["model_minus_people"] for q in Q]); out["levels_summary_" + m] = dict(mean_difference=float(d.mean()), mean_absolute_difference=float(np.abs(d).mean()), higher_than_people=int((d > 0).sum()), lower_than_people=int((d < 0).sum()),
                                                                                                                     largest=dict(question=Q[int(np.argmax(np.abs(d)))]["name"], difference=float(d[int(np.argmax(np.abs(d)))])))
    # ---- known groups: streams against videos, and the years
    def group_figures(rows=None):
        r_ = np.arange(n) if rows is None else rows; s, yr = stream[r_], year[r_] - 2020.0; g = {}
        for j, q in enumerate(Q):
            for who, v in dict(people=fits[q["id"]]["mean"][r_], jev14=QJ[r_, j], drex=QD[r_, j]).items():
                ok = np.isfinite(v); X = np.column_stack([np.ones(ok.sum()), s[ok], yr[ok]]); b = np.linalg.lstsq(X, v[ok], rcond=None)[0]; g[(q["id"], who)] = (float(v[ok][s[ok] == 1].mean() - v[ok][s[ok] == 0].mean()), float(b[2]))
        return g
    g0 = group_figures(); gb = [group_figures(rows) for rows in boots[:min(B, 500)]]; out["groups"] = dict(questions={})
    for q in Q:
        out["groups"]["questions"][q["id"]] = dict(name=q["name"], rated_before=q["rated_before"], **{who: dict(streams_minus_videos=g0[(q["id"], who)][0], streams_minus_videos_interval=P5([g[(q["id"], who)][0] for g in gb]),
                                                                                                    per_year=g0[(q["id"], who)][1], per_year_interval=P5([g[(q["id"], who)][1] for g in gb])) for who in ("people", "jev14", "drex")})
    for which, k in (("streams_minus_videos", 0), ("per_year", 1)):
        for label, qs in (("all_16", Q), ("the_12_rated_before", [q for q in Q if q["rated_before"]])):
            for m in MODELS:
                pp = lambda g: np.array([g[(q["id"], "people")][k] for q in qs]); mm = lambda g: np.array([g[(q["id"], m)][k] for q in qs])
                out["groups"].setdefault("across_questions", {}).setdefault(which, {}).setdefault(label, {})[m] = dict(correlation=corr(pp(g0), mm(g0)), interval=P5([corr(pp(g), mm(g)) for g in gb]), same_sign=int((np.sign(pp(g0)) == np.sign(mm(g0))).sum()), questions=len(qs),
                                                                                                                 slope_people_on_model=float(np.polyfit(mm(g0), pp(g0), 1)[0]))
    def group_dims(rows=None):
        r_ = np.arange(n) if rows is None else rows; s, yr = stream[r_], year[r_] - 2020.0; g = {}
        for k, f in enumerate(fids):
            for who, v in dict(people=comp[f][0][r_], jev14=FJ[r_, k], drex=FD[r_, k]).items():
                ok = np.isfinite(v); sd = v[ok].std(ddof=1); X = np.column_stack([np.ones(ok.sum()), s[ok], yr[ok]]); b = np.linalg.lstsq(X, v[ok], rcond=None)[0]; g[(f, who)] = (float((v[ok][s[ok] == 1].mean() - v[ok][s[ok] == 0].mean()) / sd), float(b[2] / sd))
        return g
    d0 = group_dims(); db = [group_dims(rows) for rows in boots[:min(B, 500)]]
    out["groups"]["dimensions"] = {f: {who: dict(streams_minus_videos_in_sd=d0[(f, who)][0], streams_minus_videos_interval=P5([g[(f, who)][0] for g in db]), per_year_in_sd=d0[(f, who)][1], per_year_interval=P5([g[(f, who)][1] for g in db])) for who in ("people", "jev14", "drex")} for f in fids}
    for which, k in (("streams_minus_videos", 0), ("per_year", 1)):
        for m in MODELS:
            pp = lambda g: np.array([g[(f, "people")][k] for f in fids]); mm = lambda g: np.array([g[(f, m)][k] for f in fids])
            out["groups"].setdefault("across_dimensions", {}).setdefault(which, {})[m] = dict(correlation=corr(pp(d0), mm(d0)), interval=P5([corr(pp(g), mm(g)) for g in db]), same_sign=int((np.sign(pp(d0)) == np.sign(mm(d0))).sum()))
    # ---- the first study, repeated
    rq = res2["questions"]; both = [q["id"] for q in Q if rq[q["id"]]["first_study"]]; out["replication"] = dict(questions={})
    for q in both:
        a, b = rq[q]["first_study"], rq[q]; out["replication"]["questions"][q] = dict(name=b["name"], first=dict(reliability=a["reliability"], jev14=a["jev14"], drex=a["drex"], verdict=a["verdict"]),
            second=dict(reliability=b["reliability"], reliability_interval=b["reliability_interval"], jev14=b["jev14_with_mean"], jev14_interval=b["jev14_interval"], drex=b["drex_with_mean"], drex_interval=b["drex_interval"], verdict=b["verdict_not_used"]),
            second_levels_in=out["sources"][q]["reliability_levels_in"], difference=dict(reliability=b["reliability"] - a["reliability"], reliability_levels_in=out["sources"][q]["reliability_levels_in"] - a["reliability"], jev14=b["jev14_with_mean"] - a["jev14"], drex=b["drex_with_mean"] - a["drex"]))
    arr = lambda k1, k2: np.array([out["replication"]["questions"][q][k1][k2] for q in both])
    out["replication"]["across_the_twelve"] = dict(questions=len(both), reliability=dict(correlation=corr(arr("first", "reliability"), arr("second", "reliability")), mean_first=float(arr("first", "reliability").mean()), mean_second=float(arr("second", "reliability").mean()),
        mean_second_levels_in=float(np.mean([out["replication"]["questions"][q]["second_levels_in"] for q in both])), largest_change=float(np.abs(arr("difference", "reliability")).max())),
        **{m: dict(correlation=corr(arr("first", m), arr("second", m)), mean_first=float(arr("first", m).mean()), mean_second=float(arr("second", m).mean()), largest_change=float(np.abs(arr("difference", m)).max())) for m in MODELS},
        same_verdict=int(sum(out["replication"]["questions"][q]["first"]["verdict"] == out["replication"]["questions"][q]["second"]["verdict"] for q in both)))
    out["replication"]["dimensions"] = {}
    for f in fids:
        a = first_dims[f]; b = res2["dimensions"][f]
        out["replication"]["dimensions"][f] = dict(first=dict(markers=a["names"], reliability=a["reliability_of_mean"], reliability_interval=a["reliability_of_mean_interval"], jev14=dict(with_mean=a["dimension_jev14"]["with_mean"], with_panel=a["dimension_jev14"]["with_panel"]),
                                                               drex=dict(with_mean=a["dimension_drex"]["with_mean"], with_panel=a["dimension_drex"]["with_panel"]), by_the_rule=a["by_the_rule_for_questions"]),
                                                   second=dict(markers=[rq[x]["name"] for x in b["questions"]], reliability=b["plain"]["reliability"], reliability_interval=b["plain"]["reliability_interval"], jev14=b["plain"]["jev14_with_mean"], drex=b["plain"]["drex_with_mean"], verdict=b["plain"]["verdict"], settled=b["plain"]["settled"]))
    # ---- a model against an unlimited panel; more raters
    out["panel"] = {}
    for f in fids:
        d = res2["dimensions"][f]; w_ = d["with_the_model"]; r5 = d["plain"]["reliability"]; ratio = (1 - r5) / r5; rk = lambda k: 1 / (1 + (5 / k) * ratio); need = lambda t: int(math.ceil(5 * ratio * t / (1 - t) - 1e-9))
        nz = 5 * w_["noise_of_mean"] / w_["var_true"]; worth = lambda rho: (rho ** 2 / (1 - rho ** 2)) * nz if rho < 0.999 else float("inf")
        out["panel"][f] = dict(jev14_with_panel=w_["jev14_with_panel"], jev14_interval=w_["jev14_with_panel_interval"], drex_with_panel=w_["drex_with_panel"], drex_interval=w_["drex_with_panel_interval"],
                               one_rating_of_each_marker_with_panel=math.sqrt(w_["var_true"] / (w_["var_true"] + 5 * w_["noise_of_mean"])), mean_of_five_with_panel=math.sqrt(w_["reliability"]),
                               raters_per_question_a_model_is_worth=dict(jev14=worth(w_["jev14_with_panel"]), drex=worth(w_["drex_with_panel"])),
                               reliability_by_raters_per_question={str(k): rk(k) for k in (1, 2, 3, 5, 8, 10, 15, 20)}, raters_per_question_for_0_60=need(0.60), raters_per_question_for_0_70=need(0.70))
    out["panel_summary"] = {m: dict(mean=float(np.mean([out["panel"][f][m + "_with_panel"] for f in fids])), lowest=float(min(out["panel"][f][m + "_with_panel"] for f in fids)), highest=float(max(out["panel"][f][m + "_with_panel"] for f in fids))) for m in MODELS}
    out["panel_summary"]["one_rating_of_each_marker"] = dict(mean=float(np.mean([out["panel"][f]["one_rating_of_each_marker_with_panel"] for f in fids])), lowest=float(min(out["panel"][f]["one_rating_of_each_marker_with_panel"] for f in fids)), highest=float(max(out["panel"][f]["one_rating_of_each_marker_with_panel"] for f in fids)))
    # ---- the raters
    qt = lambda v: dict(zip(("p05", "p25", "median", "p75", "p95"), (float(x) for x in np.nanpercentile(v, [5, 25, 50, 75, 95])))); fin = lambda v: v[np.isfinite(v)]
    out["raters"] = dict(valid=int(D["n_valid"]), **{k: dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v, ddof=1)), **qt(v)) for k, v in RT.items() if k != "ratings"}, minutes=dict(mean=D["time"]["summary"]["all"]["mean"], p25=D["time"]["summary"]["all"]["first_quartile"], median=D["time"]["summary"]["all"]["median"], p75=D["time"]["summary"]["all"]["third_quartile"]))
    out["raters"]["with_others_below_zero"] = int((fin(RT["with_others"]) < 0).sum()); out["raters"]["closer_to_the_site_model_than_to_the_other_raters"] = float(np.nanmean(RT["with_jev14"] > RT["with_others"]))
    out["raters"]["between_traits"] = dict(agreement_with_others_and_with_site_model=corr(RT["with_others"], RT["with_jev14"]), agreement_and_cannot_judge=corr(RT["with_others"], RT["cannot_judge"]))
    out["raters"]["histograms"] = dict(minutes_by_arm={ARMS[a]: dict(edges=[v[0] for v in D["time"]["histogram"][ARMS[a]]] + [D["time"]["histogram"][ARMS[a]][-1][1]], counts=[v[2] for v in D["time"]["histogram"][ARMS[a]]]) for a in (1, 0)})
    return out


# ================================================================ 7  second study: added afterwards, second round
def study2_more(F, D, res2, fu, ctx, first, first_further, B=1000, seed=20261124):
    Q, P, ex, idx, pairs, strata, fids = F["Q"], F["P"], F["ex"], F["idx"], F["pairs"], F["strata"], F["fids"]; n = len(ex); n_sets = sum(len(v) for v in F["sets"].values()); kr, fits, comp = ctx["kr"], ctx["fits"], ctx["comp"]
    rng = np.random.default_rng(seed); boots = [pairs[np.concatenate([rng.choice(v, len(v)) for v in strata])].ravel() for _ in range(B)]
    QJ, QD, FJ, FD = P["QJ"][idx], P["QD"][idx], P["FJ"][idx], P["FD"][idx]; MF = dict(jev14=FJ, drex=FD); qcol = {q["id"]: j for j, q in enumerate(Q)}; okr = np.isfinite(D["y"])
    kind = F["kind"]; year = np.array([e["year"] for e in ex], float); stream = (kind == "stream").astype(float); out = dict(valid_raters=int(D["n_valid"]), resamples=B)
    def figs(cmp, k):
        o = {}
        for j, f in enumerate(fids):
            h, c = cmp[f]; m = np.isfinite(h); pl = plain(h[m], c[m], float(np.mean(c[m])) * k, np.column_stack([FJ[m, j], FD[m, j]])); o[f] = (pl["reliability"], pl["jev14"], pl["drex"])
        return o
    # ---- raters: each rater left out in turn (jackknife), beside the resampling of recordings
    base = figs(comp, kr); bt = {f: [] for f in fids}
    for rows in boots:
        for j, f in enumerate(fids):
            h, c = comp[f][0][rows], comp[f][1][rows]; m = np.isfinite(h); pl = plain(h[m], c[m], float(np.mean(c[m])) * kr, np.column_stack([FJ[rows][m, j], FD[rows][m, j]])); bt[f].append((pl["reliability"], pl["jev14"], pl["drex"]))
    bt = {f: np.array(v) for f, v in bt.items()}; jk = {f: [] for f in fids}; krj = (D["n_valid"] - 1) / n_sets; form_q = {x: [q for q in Q if q["form"] == x] for x in "AB"}
    for r in range(D["n_valid"]):
        fr = dict(fits); fr.update(fits_of(D, F, drop=r, only=form_q[D["form"][r]])); fg = figs(composites(fr, Q, fids), krj)
        for f in fids: jk[f].append(fg[f])
    out["raters"] = dict(dimensions={})
    for f in fids:
        J = np.array(jk[f]); nj = len(J); se_r = np.sqrt((nj - 1) / nj * ((J - J.mean(0)) ** 2).sum(0)); se_b = bt[f].std(0, ddof=1); est = np.array(base[f]); fac = np.sqrt(1 + (se_r / se_b) ** 2)
        pl_ = res2["dimensions"][f]["plain"]; iv = np.array([pl_["reliability_interval"], pl_["jev14_interval"], pl_["drex_interval"]]); wide = np.clip(np.column_stack([est - fac * (est - iv[:, 0]), est + fac * (iv[:, 1] - est)]), 0, 1)
        rc = lambda I: [max(I[1][0], I[2][0]), max(I[1][1], I[2][1])]; point = verdict(est[0], max(est[1], est[2]))
        out["raters"]["dimensions"][f] = dict(verdict=point, **{nm: dict(estimate=float(est[k]), se_recordings=float(se_b[k]), se_raters=float(se_r[k]), ratio=float(se_r[k] / se_b[k]), interval_recordings=[float(x) for x in iv[k]], interval_both=[float(x) for x in wide[k]],
                                                                         largest_change_without_one_rater=float(np.abs(J[:, k] - est[k]).max())) for k, nm in enumerate(("reliability", "jev14", "drex"))},
                                              settled_recordings=bool(settled(point, list(iv[0]), rc(iv))), settled_both=bool(settled(point, list(wide[0]), rc(wide))),
                                              verdicts_without_one_rater={v: int(c) for v, c in collections.Counter(verdict(a, max(b_, c_)) for a, b_, c_ in J).most_common()})
    # ---- rounds: the five rounds of a form as five panels of one rater per excerpt; no model
    rnd = D["round"]
    def round_matrix(qid):
        S_ = np.zeros((n, 5)); N_ = np.zeros((n, 5)); m = (D["qid"] == qid) & okr; np.add.at(S_, (D["item"][m], rnd[D["rater"][m]]), D["y"][m]); np.add.at(N_, (D["item"][m], rnd[D["rater"][m]]), 1)
        with np.errstate(invalid="ignore", divide="ignore"): return np.where(N_ > 0, S_ / N_, np.nan)
    def icc(Y):
        m = np.isfinite(Y); k = m.sum(1); keep = k >= 2; Y, m, k = Y[keep], m[keep], k[keep]; mean = np.nansum(Y, 1) / k; N_ = k.sum(); g = len(k); grand = np.nansum(Y) / N_
        msb = float((k * (mean - grand) ** 2).sum() / (g - 1)); msw = float(np.nansum((Y - mean[:, None]) ** 2) / (N_ - g)); k0 = (N_ - (k ** 2).sum() / N_) / (g - 1); vt = max((msb - msw) / k0, 0.0); return vt / (vt + msw) if vt + msw > 0 else 0.0
    RM = {q["id"]: round_matrix(q["id"]) for q in Q}; out["rounds"] = dict(questions={}, dimensions={})
    for q in Q: r1 = icc(RM[q["id"]]); out["rounds"]["questions"][q["id"]] = dict(name=q["name"], one_rating=r1, mean_of_five=sb(r1), model_levels_in=fu["sources"][q["id"]]["reliability_levels_in"], model_levels_out=fu["sources"][q["id"]]["reliability_levels_out"])
    for f in fids:
        qa, qb = [q for q in Q if q["dimension"] == f]; Y = (qa["sign"] * RM[qa["id"]] + qb["sign"] * RM[qb["id"]]) / 2; r1 = icc(Y); bs = [sb(icc(Y[rows])) for rows in boots[:min(B, 500)]]
        out["rounds"]["dimensions"][f] = dict(one_rating_of_each_marker=r1, five_raters_per_question=sb(r1), interval=P5(bs), model_levels_in=fu["sources_dimensions"][f]["reliability_levels_in"], model_levels_out=fu["sources_dimensions"][f]["reliability_levels_out"])
    # ---- pooled: the twelve questions of both studies (levels left in on both sides)
    se_of = lambda iv: (iv[1] - iv[0]) / 3.92; out["pooled"] = dict(questions={})
    for q in Q:
        if not q["rated_before"]: continue
        ft = fits[q["id"]]; j = qcol[q["id"]]; vl, vr = float(ft["var_level"]), float(ft["var_rest"])
        def second(rows=None):
            h, c = (ft["mean"], ft["noise"]) if rows is None else (ft["mean"][rows], ft["noise"][rows]); XJ_, XD_ = (QJ[:, j], QD[:, j]) if rows is None else (QJ[rows, j], QD[rows, j]); ok = np.isfinite(h); cb = float(np.mean(c[ok])); dv = max(float(np.var(h[ok], ddof=1)) - cb, 1e-12)
            o = dict(reliability=dv / (dv + (vl + vr) / 5))
            for m, X in (("jev14", XJ_), ("drex", XD_)): r = corr(X[ok], h[ok]); o[m] = min(r * math.sqrt((dv + cb) / (dv + (vl + vr) / 5)), 1.0); o[m + "_with_panel"] = min(r * math.sqrt((dv + cb) / dv), 1.0)
            return o
        s2 = second(); b2 = [second(rows) for rows in boots]; R1 = first[q["id"]]["rebuilt"]; P1 = first_further["questions"][q["id"]]["panel"]
        fst = dict(reliability=(R1["reliability_of_mean"], se_of(R1["reliability_of_mean_interval"])), jev14=(R1["jev14"]["correlation"], se_of(R1["jev14"]["interval"])), drex=(R1["drex"]["correlation"], se_of(R1["drex"]["interval"])),
                   jev14_with_panel=(P1["jev14"]["correlation"], se_of(P1["jev14"]["interval"])), drex_with_panel=(P1["drex"]["correlation"], se_of(P1["drex"]["interval"])))
        o = dict(name=q["name"], dimension=q["dimension"])
        for k, (e1, s1) in fst.items():
            e2 = s2[k]; sd2 = float(np.std([b[k] for b in b2], ddof=1)); w1, w2 = 1 / s1 ** 2, 1 / max(sd2, 1e-6) ** 2; pe = (w1 * e1 + w2 * e2) / (w1 + w2); ps = math.sqrt(1 / (w1 + w2))
            o[k] = dict(first=e1, first_se=s1, second=e2, second_se=sd2, pooled=pe, pooled_se=ps, interval=[pe - 1.96 * ps, min(pe + 1.96 * ps, 1.0)], z_of_the_difference=(e2 - e1) / math.sqrt(s1 ** 2 + sd2 ** 2))
        rel = o["reliability"]; best = max(MODELS, key=lambda m: o[m]["pooled"]); rint = [max(o["jev14"]["interval"][0], o["drex"]["interval"][0]), max(o["jev14"]["interval"][1], o["drex"]["interval"][1])]
        o["verdict_pooled"] = verdict(rel["pooled"], o[best]["pooled"]); o["settled_pooled"] = bool(settled(o["verdict_pooled"], rel["interval"], rint)); o["verdict_first"] = first[q["id"]]["verdict"]; out["pooled"]["questions"][q["id"]] = o
    pq = out["pooled"]["questions"]; out["pooled"]["summary"] = dict(usable=[v["name"] for v in pq.values() if v["verdict_pooled"] == "usable"], usable_and_settled=[v["name"] for v in pq.values() if v["verdict_pooled"] == "usable" and v["settled_pooled"]],
        differ_beyond_chance={k: [v["name"] for v in pq.values() if abs(v[k]["z_of_the_difference"]) > 1.96] for k in ("reliability", "jev14", "drex")},
        mean_with_panel={m: dict(first=float(np.mean([v[m + "_with_panel"]["first"] for v in pq.values()])), second=float(np.mean([v[m + "_with_panel"]["second"] for v in pq.values()])), pooled=float(np.mean([v[m + "_with_panel"]["pooled"] for v in pq.values()]))) for m in MODELS})
    # ---- ordinal: alpha with two metrics; the primary figures on normal scores
    out["ordinal"] = dict(questions={}, dimensions={}); fits_n = {}
    for q in Q:
        m = (D["qid"] == q["id"]) & okr; units = collections.defaultdict(list)
        for i, y in zip(D["item"][m], D["y"][m]): units[int(i)].append(int(y))
        ai, ao = kripp(units.values(), "interval"), kripp(units.values(), "ordinal")
        out["ordinal"]["questions"][q["id"]] = dict(name=q["name"], alpha_interval=ai, alpha_ordinal=ao, mean_of_five_interval=sb(max(ai, 0)), mean_of_five_ordinal=sb(max(ao, 0)), model_levels_in=fu["sources"][q["id"]]["reliability_levels_in"])
        y = D["y"][m].astype(int); cnt = np.bincount(y, minlength=5); cum = np.cumsum(cnt); mid = (cum - cnt / 2) / cum[-1]; z = stats.norm.ppf(np.clip(mid, 1e-6, 1 - 1e-6)); W = (D["arm"][D["rater"][m]] - 0.5)[:, None]
        fits_n[q["id"]] = fit_crossed(z[y], D["rater"][m], D["item"][m], n, W); fits_n[q["id"]]["ratings_per_excerpt"] = np.bincount(D["item"][m], minlength=n)
    fn = figs(composites(fits_n, Q, fids), kr)
    for j, f in enumerate(fids):
        h = comp[f][0]; ok = np.isfinite(h); out["ordinal"]["dimensions"][f] = dict(reliability=base[f][0], reliability_normal_scores=fn[f][0], jev14=base[f][1], jev14_normal_scores=fn[f][1], drex=base[f][2], drex_normal_scores=fn[f][2],
            jev14_rank_correlation=float(stats.spearmanr(FJ[ok, j], h[ok])[0]), drex_rank_correlation=float(stats.spearmanr(FD[ok, j], h[ok])[0]), verdict=verdict(base[f][0], max(base[f][1], base[f][2])), verdict_normal_scores=verdict(fn[f][0], max(fn[f][1], fn[f][2])))
    od = out["ordinal"]["dimensions"]; out["ordinal"]["largest_change"] = dict(reliability=float(max(abs(v["reliability_normal_scores"] - v["reliability"]) for v in od.values())), correlation=float(max(max(abs(v["jev14_normal_scores"] - v["jev14"]), abs(v["drex_normal_scores"] - v["drex"])) for v in od.values())),
                                                                              verdicts_that_change=[f for f, v in od.items() if v["verdict"] != v["verdict_normal_scores"]])
    # ---- within: the two passages of a recording
    out["within"] = dict(dimensions={})
    def within2(f, j, pr):
        h, c = comp[f]; a, b = pr[:, 0], pr[:, 1]; ok = np.isfinite(h[a]) & np.isfinite(h[b]); a, b = a[ok], b[ok]; dH = h[a] - h[b]; mH = (h[a] + h[b]) / 2; nd = float(np.mean(c[a] + c[b])); o = {}
        vd = float(np.var(dH, ddof=1)); vm = float(np.var(mH, ddof=1)); td = max(vd - nd, 1e-12); tm = max(vm - nd / 4, 1e-12); o["share_of_true_differences_within_recordings"] = (td / 2) / (td / 2 + max(tm - td / 4, 1e-12))
        o["reliability_of_the_difference"] = td / vd; o["reliability_of_the_recording_mean"] = tm / vm
        for m, X in MF.items(): rw = corr(X[a, j] - X[b, j], dH); rb = corr((X[a, j] + X[b, j]) / 2, mH); o[m] = dict(within=rw, within_with_panel=min(rw * math.sqrt(vd / td), 1.0), between=rb, between_with_panel=min(rb * math.sqrt(vm / tm), 1.0))
        return o
    for j, f in enumerate(fids):
        o = within2(f, j, pairs); bs = [within2(f, j, pairs[np.concatenate([rng.choice(v, len(v)) for v in strata])]) for _ in range(min(B, 1000))]
        for m in MODELS:
            for k in ("within", "within_with_panel", "between", "between_with_panel"): o[m][k + "_interval"] = P5([b[m][k] for b in bs])
            o[m]["within_minus_between_with_panel"] = o[m]["within_with_panel"] - o[m]["between_with_panel"]; o[m]["within_minus_between_interval"] = P5([b[m]["within_with_panel"] - b[m]["between_with_panel"] for b in bs])
        out["within"]["dimensions"][f] = o
    wd = out["within"]["dimensions"]; out["within"]["summary"] = {m: dict(mean_within=float(np.mean([wd[f][m]["within"] for f in fids])), mean_between=float(np.mean([wd[f][m]["between"] for f in fids])), mean_within_with_panel=float(np.mean([wd[f][m]["within_with_panel"] for f in fids])),
                                                                         mean_between_with_panel=float(np.mean([wd[f][m]["between_with_panel"] for f in fids]))) for m in MODELS}
    # ---- halo: what is left of one rater's ratings of one excerpt on different questions
    E = {}
    for q in Q:
        ft = fits[q["id"]]; m = (D["qid"] == q["id"]) & okr & np.isfinite(ft["mean"][D["item"]]); res = D["y"][m] - ft["mean"][D["item"][m]] - ft["fixed"][0] * (D["arm"][D["rater"][m]] - 0.5); lam = ft["var_level"] / ft["var_rest"]
        rs = np.bincount(D["rater"][m], weights=res, minlength=D["n_valid"]); rn = np.bincount(D["rater"][m], minlength=D["n_valid"]); u = np.where(rn > 0, lam * rs / (1 + rn * lam), 0.0); e = res - u[D["rater"][m]]
        E[q["id"]] = {(int(r), int(i)): float(v) for r, i, v in zip(D["rater"][m], D["item"][m], e)}
    out["halo"] = dict(forms={}); resid_cov = {}
    for x in "AB":
        qs = form_q[x]; C = np.full((len(qs), len(qs)), np.nan)
        for a in range(len(qs)):
            for b_ in range(a + 1, len(qs)):
                ka = sorted(set(E[qs[a]["id"]]) & set(E[qs[b_]["id"]])); va = np.array([qs[a]["sign"] * E[qs[a]["id"]][k] for k in ka]); vb = np.array([qs[b_]["sign"] * E[qs[b_]["id"]][k] for k in ka]); C[a, b_] = C[b_, a] = corr(va, vb)
                resid_cov[(qs[a]["dimension"], qs[b_]["dimension"], x)] = resid_cov[(qs[b_]["dimension"], qs[a]["dimension"], x)] = float(np.mean(va * vb))
        iu = np.triu_indices(len(qs), 1); order = np.argsort(-np.abs(C[iu])); out["halo"]["forms"][x] = dict(questions=[q["name"] for q in qs], mean=float(np.mean(C[iu])), mean_absolute=float(np.mean(np.abs(C[iu]))), lowest=float(C[iu].min()), highest=float(C[iu].max()),
            largest=[dict(questions=[qs[iu[0][k]]["name"], qs[iu[1][k]]["name"]], correlation=float(C[iu][k])) for k in order[:5]])
    H = np.column_stack([comp[f][0] for f in fids]); nf = len(fids); tv = np.array([max(float(np.nanvar(H[:, k], ddof=1)) - float(np.nanmean(comp[f][1])), 1e-12) for k, f in enumerate(fids)]); kbar = D["n_valid"] / n_sets * 5
    Cobs = np.full((nf, nf), np.nan); Ctrue = np.full((nf, nf), np.nan)
    for a in range(nf):
        for b_ in range(a + 1, nf):
            ok = np.isfinite(H[:, a]) & np.isfinite(H[:, b_]); cv = float(np.cov(H[ok, a], H[ok, b_])[0, 1]); same = sum(resid_cov.get((fids[a], fids[b_], x), 0.0) for x in "AB") / (4 * kbar)
            Cobs[a, b_] = Cobs[b_, a] = corr(H[:, a], H[:, b_]); Ctrue[a, b_] = Ctrue[b_, a] = float(np.clip((cv - same) / math.sqrt(tv[a] * tv[b_]), -1, 1))
    iu = np.triu_indices(nf, 1); out["halo"]["dimensions"] = dict(order=list(fids),
        largest=[dict(dimensions=[fids[iu[0][k]], fids[iu[1][k]]], as_observed=float(Cobs[iu][k]), without_noise_and_same_rater_part=float(Ctrue[iu][k]), jev14=float(np.corrcoef(FJ.T)[iu][k]), drex=float(np.corrcoef(FD.T)[iu][k])) for k in np.argsort(-np.abs(Ctrue[iu]))[:8]],
        pattern_with_models={m: dict(as_observed=corr(np.corrcoef(MF[m].T)[iu], Cobs[iu]), without_noise_and_same_rater_part=corr(np.corrcoef(MF[m].T)[iu], Ctrue[iu])) for m in MODELS})
    # ---- bias: at the same model score, streams against videos, and the year
    out["bias"] = dict(dimensions={})
    def bias(f, j, m, rows=None):
        r_ = np.arange(n) if rows is None else rows; h = comp[f][0][r_]; ok = np.isfinite(h); X = np.column_stack([np.ones(ok.sum()), MF[m][r_, j][ok], stream[r_][ok], year[r_][ok] - 2020.0]); b = np.linalg.lstsq(X, h[ok], rcond=None)[0]; sd = h[ok].std(ddof=1)
        return float(b[2] / sd), float(b[3] / sd)
    for j, f in enumerate(fids):
        o = {}
        for m in MODELS: e = bias(f, j, m); bs = np.array([bias(f, j, m, rows) for rows in boots[:min(B, 1000)]]); o[m] = dict(streams=e[0], streams_interval=P5(bs[:, 0]), per_year=e[1], per_year_interval=P5(bs[:, 1]))
        out["bias"]["dimensions"][f] = o
    # ---- both: the two models' dimension scores averaged
    out["both"] = dict(dimensions={}); zs = lambda v: (v - v.mean()) / v.std()
    def both(f, j, rows=None):
        r_ = np.arange(n) if rows is None else rows; h, c = comp[f][0][r_], comp[f][1][r_]; ok = np.isfinite(h); h, c = h[ok], c[ok]; cb = float(np.mean(c)); dv = max(float(np.var(h, ddof=1)) - cb, 1e-12); att = math.sqrt((dv + cb) / (dv + cb * kr))
        a, b_ = FJ[r_, j][ok], FD[r_, j][ok]; return min(corr(zs(a) + zs(b_), h) * att, 1.0), min(corr(a, h) * att, 1.0), min(corr(b_, h) * att, 1.0)
    for j, f in enumerate(fids):
        e = both(f, j); bs = np.array([both(f, j, rows) for rows in boots]); out["both"]["dimensions"][f] = dict(both=e[0], both_interval=P5(bs[:, 0]), jev14=e[1], drex=e[2], both_minus_better=e[0] - max(e[1], e[2]), both_minus_better_interval=P5(bs[:, 0] - np.maximum(bs[:, 1], bs[:, 2])),
                                                                                                           both_minus_jev14=e[0] - e[1], both_minus_jev14_interval=P5(bs[:, 0] - bs[:, 1]))
    bo = out["both"]["dimensions"]; BS = np.array([[both(f, j, rows) for j, f in enumerate(fids)] for rows in boots[:min(B, 500)]])
    out["both"]["over_the_eight"] = dict(both=float(np.mean([bo[f]["both"] for f in fids])), jev14=float(np.mean([bo[f]["jev14"] for f in fids])), drex=float(np.mean([bo[f]["drex"] for f in fids])), both_minus_jev14=float(np.mean([bo[f]["both_minus_jev14"] for f in fids])),
                                         both_minus_jev14_interval=P5((BS[:, :, 0] - BS[:, :, 1]).mean(1)))
    # ---- misses: do the two models miss alike (rater noise taken out of both)
    out["misses"] = dict(dimensions={})
    for j, f in enumerate(fids):
        h, c = comp[f]; ok = np.isfinite(h); res = {}
        for m in MODELS: b = np.polyfit(MF[m][ok, j], h[ok], 1); res[m] = h[ok] - np.polyval(b, MF[m][ok, j])
        cb = float(np.mean(c[ok])); cv = float(np.cov(res["jev14"], res["drex"])[0, 1]); vj, vd = float(np.var(res["jev14"], ddof=1)), float(np.var(res["drex"], ddof=1))
        out["misses"]["dimensions"][f] = dict(shared_as_observed=corr(res["jev14"], res["drex"]), shared=float(np.clip((cv - cb) / math.sqrt(max(vj - cb, 1e-12) * max(vd - cb, 1e-12)), -1, 1)) if vj > cb and vd > cb else None)
    # ---- order: the question shown first; fatigue: over the eight excerpts of a session
    shift = D["shift"]; qpos = np.array([(F["forms"][D["form"][r]].index(qq) - shift[r]) % 8 for r, qq in zip(D["rater"], D["qid"])]); firstq = qpos == 0; eff = []; out["order"] = dict(questions={})
    for q in Q:
        m = (D["qid"] == q["id"]) & okr; W = np.column_stack([D["arm"][D["rater"][m]] - 0.5, firstq[m].astype(float)]); ft = fit_crossed(D["y"][m], D["rater"][m], D["item"][m], n, W)
        out["order"]["questions"][q["id"]] = dict(name=q["name"], level_when_first=float(ft["fixed"][1]), se=float(ft["fixed_se"][1])); eff.append((float(ft["fixed"][1]), float(ft["fixed_se"][1])))
    eff = np.array(eff); w = 1 / eff[:, 1] ** 2; out["order"]["level_when_first"] = dict(mean=float((w * eff[:, 0]).sum() / w.sum()), se=float(math.sqrt(1 / w.sum())), largest=float(np.abs(eff[:, 0]).max()), beyond_two_se=int((np.abs(eff[:, 0] / eff[:, 1]) > 2).sum()), questions=len(eff))
    z2 = np.full(len(D["y"]), np.nan)             # the squared distance of a rating from the excerpt's mean and the rater's level, in units of the question's residual variance
    for q in Q: m = np.flatnonzero((D["qid"] == q["id"]) & okr); v = np.array([E[q["id"]].get((int(r), int(i)), np.nan) for r, i in zip(D["rater"][m], D["item"][m])]); z2[m] = v ** 2 / fits[q["id"]]["var_rest"]
    def by_rater(mask): mk = mask & np.isfinite(z2); return np.bincount(D["rater"][mk], weights=z2[mk], minlength=D["n_valid"]), np.bincount(D["rater"][mk], minlength=D["n_valid"])
    def ratio_ci(mask_a, mask_b):
        sa, ca = by_rater(mask_a); sb_, cb_ = by_rater(mask_b); bs = []
        for _ in range(1000): k = rng.integers(0, D["n_valid"], D["n_valid"]); bs.append((sa[k].sum() / max(ca[k].sum(), 1)) / (sb_[k].sum() / max(cb_[k].sum(), 1)))
        return dict(first=float(sa.sum() / ca.sum()), second=float(sb_.sum() / cb_.sum()), ratio=float((sa.sum() / ca.sum()) / (sb_.sum() / cb_.sum())), interval=P5(bs))
    out["order"]["noise_first_against_later"] = ratio_ci(firstq, ~firstq); pos = D["position"]; pmin = int(pos.min()); out["fatigue"] = dict(by_position={})
    for p in range(pmin, pmin + 8): m = pos == p; s, c = by_rater(m); out["fatigue"]["by_position"][str(p - pmin + 1)] = dict(noise=float(s.sum() / c.sum()), cannot_judge=float((m & ~okr).sum() / m.sum()))
    out["fatigue"]["noise_last_four_against_first_four"] = ratio_ci(pos >= pmin + 4, pos < pmin + 4)
    return out


# ================================================================ 8  a third model on both studies
M3 = ("jev14", "drex", "d12u")


def third_model(S, F, D, ctx, first, rp, B=1000):
    """d1.2-ultra beside the two models of the analysis, against people's ratings of both studies. Its dimension scores
    are computed as those of drex-max: the scorer fitted on jev-1.14.0's ratings, applied to the model's own ratings
    standardised within the model. Added after both studies had been analysed."""
    Q, P, ex, idx, pairs, strata, fids = F["Q"], F["P"], F["ex"], F["idx"], F["pairs"], F["strata"], F["fids"]; n = len(ex); kr, fits, comp = ctx["kr"], ctx["fits"], ctx["comp"]; qids = [q["id"] for q in Q]; w = P["w"]
    QS = dict(jev14=P["QJ"], drex=P["QD"], d12u=P["QL"]); FS = dict(jev14=P["FJ"], drex=P["FD"], d12u=P["FL"]); out = dict(pool=dict(candidates=len(P["cand"])))
    def wcorr(a, b): ww = w / w.sum(); ma, mb = (a * ww).sum(), (b * ww).sum(); return float(((a - ma) * (b - mb) * ww).sum() / math.sqrt((((a - ma) ** 2) * ww).sum() * (((b - mb) ** 2) * ww).sum()))
    out["among_models"] = dict(questions={}, dimensions={})
    for a_, b_ in (("jev14", "drex"), ("jev14", "d12u"), ("drex", "d12u")):
        qv = [wcorr(QS[a_][:, j], QS[b_][:, j]) for j in range(len(Q))]; dv = [wcorr(FS[a_][:, k], FS[b_][:, k]) for k in range(len(fids))]
        out["among_models"]["questions"]["%s_%s" % (a_, b_)] = dict(median=float(np.median(qv)), lowest=float(min(qv)), highest=float(max(qv))); out["among_models"]["dimensions"]["%s_%s" % (a_, b_)] = dict(median=float(np.median(dv)), lowest=float(min(dv)), highest=float(max(dv)))
    # ---- second study
    rng = np.random.default_rng(20261122); picks = [np.concatenate([rng.choice(v, len(v)) for v in strata]) for _ in range(B)]; year = np.array([e["year"] for e in ex], float); stream = (F["kind"] == "stream").astype(float)
    def plain3(h, c, X):
        m = np.isfinite(h); h, c, X = h[m], c[m], X[m]; cb = float(np.mean(c)); dv = max(float(np.var(h, ddof=1)) - cb, 1e-12); att = math.sqrt((dv + cb) / (dv + cb * kr)); r = [corr(X[:, j], h) for j in range(X.shape[1])]
        return dv / (dv + cb * kr), [min(x * att, 1.0) for x in r], [min(x * math.sqrt((dv + cb) / dv), 1.0) for x in r]
    def block(get_h, get_X, keys):
        res, draws = {}, {}
        for j, key in enumerate(keys):
            h, c = get_h(key); X = get_X(j); rel, r, rho = plain3(h, c, X); bs = []
            for pick in picks: rows = pairs[pick].ravel(); bs.append(plain3(h[rows], c[rows], X[rows]))
            br = np.array([b[1] for b in bs]); bp = np.array([b[2] for b in bs]); draws[key] = br
            res[key] = dict(reliability=rel, **{m: dict(with_mean=r[i], interval=P5(br[:, i]), with_panel=rho[i], with_panel_interval=P5(bp[:, i])) for i, m in enumerate(M3)},
                            d12u_minus_jev14=dict(difference=r[2] - r[0], interval=P5(br[:, 2] - br[:, 0])), d12u_minus_drex=dict(difference=r[2] - r[1], interval=P5(br[:, 2] - br[:, 1])),
                            best=M3[int(np.argmax(r))], verdict_with_each={m: verdict(rel, r[i]) for i, m in enumerate(M3)})
        allr = np.array([[res[k][m]["with_mean"] for m in M3] for k in keys]); alld = np.array([draws[k] for k in keys])
        over = {m: dict(mean=float(allr[:, i].mean()), interval=P5(alld[:, :, i].mean(0))) for i, m in enumerate(M3)}
        over["d12u_minus_jev14"] = dict(difference=float((allr[:, 2] - allr[:, 0]).mean()), interval=P5((alld[:, :, 2] - alld[:, :, 0]).mean(0))); over["d12u_minus_drex"] = dict(difference=float((allr[:, 2] - allr[:, 1]).mean()), interval=P5((alld[:, :, 2] - alld[:, :, 1]).mean(0)))
        over["best_on"] = dict(collections.Counter(res[k]["best"] for k in keys)); over["with_panel"] = {m: float(np.mean([res[k][m]["with_panel"] for k in keys])) for m in M3}
        return res, over
    dims, dims_over = block(lambda f: comp[f], lambda k: np.column_stack([FS[m][idx, k] for m in M3]), fids)
    qs, qs_over = block(lambda q: (fits[q]["mean"], fits[q]["noise"]), lambda j: np.column_stack([QS[m][idx, j] for m in M3]), qids)
    s2 = dict(dimensions=dims, over_the_eight_dimensions=dims_over, questions=qs, over_the_sixteen_questions=qs_over, within_recordings={}, same_score_in_streams={}, misses_shared={}, levels={}); p0, p1 = pairs[:, 0], pairs[:, 1]
    for k, f in enumerate(fids):
        h, c = comp[f]; okp = np.isfinite(h[p0]) & np.isfinite(h[p1]); a, b = p0[okp], p1[okp]; dH = h[a] - h[b]; nd = float(np.mean(c[a] + c[b])); vd = float(np.var(dH, ddof=1)); td = max(vd - nd, 1e-12)
        s2["within_recordings"][f] = {m: dict(as_observed=corr(FS[m][idx[a], k] - FS[m][idx[b], k], dH), with_panel=min(corr(FS[m][idx[a], k] - FS[m][idx[b], k], dH) * math.sqrt(vd / td), 1.0)) for m in M3}
        o = {}; m_ok = np.isfinite(h); res = {}
        for m in M3:
            def bias(rows=None):
                r_ = np.arange(n) if rows is None else rows; hh = h[r_]; g = np.isfinite(hh); X = np.column_stack([np.ones(g.sum()), FS[m][idx[r_], k][g], stream[r_][g], year[r_][g] - 2020.0]); return float(np.linalg.lstsq(X, hh[g], rcond=None)[0][2] / hh[g].std(ddof=1))
            o[m] = dict(streams=bias(), interval=P5([bias(pairs[pick].ravel()) for pick in picks[:500]])); bb = np.polyfit(FS[m][idx[m_ok], k], h[m_ok], 1); res[m] = h[m_ok] - np.polyval(bb, FS[m][idx[m_ok], k])
        s2["same_score_in_streams"][f] = o; cb = float(np.mean(c[m_ok]))
        def shared(x, y): cv = float(np.cov(res[x], res[y])[0, 1]); vx, vy = float(np.var(res[x], ddof=1)), float(np.var(res[y], ddof=1)); return float(np.clip((cv - cb) / math.sqrt(max(vx - cb, 1e-12) * max(vy - cb, 1e-12)), -1, 1))
        s2["misses_shared"][f] = dict(d12u_jev14=shared("d12u", "jev14"), d12u_drex=shared("d12u", "drex"), jev14_drex=shared("jev14", "drex"))
    for j, q in enumerate(Q):
        h = fits[q["id"]]["mean"]; g = np.isfinite(h); s2["levels"][q["id"]] = dict(people=float(h[g].mean()), **{m: dict(model_minus_people=float(QS[m][idx[g], j].mean() - h[g].mean()), slope=float(np.polyfit(QS[m][idx[g], j], h[g], 1)[0])) for m in M3})
    s2["levels_summary"] = {m: dict(mean_absolute_difference=float(np.mean([abs(v[m]["model_minus_people"]) for v in s2["levels"].values()])), higher_than_people=int(sum(v[m]["model_minus_people"] > 0 for v in s2["levels"].values())),
                                    largest=list(max(((q["name"], s2["levels"][q["id"]][m]["model_minus_people"]) for q in Q), key=lambda x: abs(x[1])))) for m in M3}
    gp = lambda v: float(v[stream == 1][np.isfinite(v[stream == 1])].mean() - v[stream == 0][np.isfinite(v[stream == 0])].mean()); pg = np.array([gp(fits[q]["mean"]) for q in qids])
    s2["known_groups_streams_minus_videos"] = {m: dict(correlation_across_16_questions=corr(pg, [gp(QS[m][idx, j]) for j in range(len(Q))])) for m in M3}
    s2["four_calls"] = {m: dict(middle_spread=float(np.median([v["within"] for v in rp[m]["questions"].values()])), middle_reliability=float(np.median([v["reliability_of_one_call"] for v in rp[m]["questions"].values()])), lowest_reliability=float(min(v["reliability_of_one_call"] for v in rp[m]["questions"].values())),
                                dimensions=[float(min(v["reliability_of_one_call"] for v in rp[m]["dimensions"].values())), float(max(v["reliability_of_one_call"] for v in rp[m]["dimensions"].values()))]) for m in M3}
    out["second_study"] = s2
    # ---- first study: rebuilt for the population from a regression on all three models' scores; the random part alone
    Q1 = S["Q"]; ex1 = S["ex"]; st = collections.defaultdict(list)
    for i, e in enumerate(ex1): st[(S["meta"][e]["part"], S["meta"][e]["kind"])].append(i)
    st1 = [np.array(v) for v in st.values()]; rng1 = np.random.default_rng(20261125); draws = [np.concatenate([rng1.choice(v, len(v)) for v in st1]) for _ in range(B)]; rnd = S["random"]; fq1 = {}; keep = {}
    for q in Q1:
        j = qids.index(q); groups = S["groups"][q]; hm = np.array([g.mean() for g in groups]); Xp = np.column_stack([QS[m][:, j] for m in M3]); Sig, mu = wcov(Xp, w); X = Xp[S["idx"]]; combos = {m: list(np.eye(3)[i]) for i, m in enumerate(M3)}
        def one(ii):
            g = [groups[i] for i in ii]; msw, inv_n = within(g); r = rebuild(hm[ii], msw, inv_n, X[ii], Sig, mu, combos); rr = np.flatnonzero(rnd[ii]); return r, [float(stats.spearmanr(X[ii][rr, c_], hm[ii][rr])[0]) for c_ in range(3)]
        est, rk = one(np.arange(len(ex1))); bs = [one(ii) for ii in draws]; wm = np.array([[b[0][m]["with_mean"] for m in M3] for b in bs]); rb = np.array([b[1] for b in bs]); keep[q] = wm
        fq1[q] = dict(name=S["name"][q], reliability_of_mean=est["reliability_mean"], as_fixed_in_advance=dict(reliability_of_mean=first[q]["rebuilt"]["reliability_of_mean"], jev14=first[q]["rebuilt"]["jev14"]["correlation"], drex=first[q]["rebuilt"]["drex"]["correlation"]),
            **{m: dict(with_mean=est[m]["with_mean"], interval=P5(wm[:, i]), with_panel=est[m]["with_panel"], random_part_rank_correlation=rk[i], random_part_interval=P5(rb[:, i]), minus_people=est[m]["minus_people"]) for i, m in enumerate(M3)},
            d12u_minus_jev14=dict(difference=est["d12u"]["with_mean"] - est["jev14"]["with_mean"], interval=P5(wm[:, 2] - wm[:, 0])), d12u_minus_drex=dict(difference=est["d12u"]["with_mean"] - est["drex"]["with_mean"], interval=P5(wm[:, 2] - wm[:, 1])),
            best=max(M3, key=lambda m: est[m]["with_mean"]))
    W = np.array([keep[q] for q in Q1]); E1 = np.array([[fq1[q][m]["with_mean"] for m in M3] for q in Q1])
    out["first_study"] = dict(questions=fq1, over_the_twelve_questions=dict(**{m: dict(mean=float(E1[:, i].mean()), interval=P5(W[:, :, i].mean(0)), with_panel=float(np.mean([fq1[q][m]["with_panel"] for q in Q1])), random_part=float(np.mean([fq1[q][m]["random_part_rank_correlation"] for q in Q1]))) for i, m in enumerate(M3)},
        d12u_minus_jev14=dict(difference=float((E1[:, 2] - E1[:, 0]).mean()), interval=P5((W[:, :, 2] - W[:, :, 0]).mean(0))), d12u_minus_drex=dict(difference=float((E1[:, 2] - E1[:, 1]).mean()), interval=P5((W[:, :, 2] - W[:, :, 1]).mean(0))), best_on=dict(collections.Counter(fq1[q]["best"] for q in Q1)),
        largest_change_of_the_two_earlier_models_against_the_figures_fixed_in_advance=float(max(abs(fq1[q][m]["with_mean"] - fq1[q]["as_fixed_in_advance"][m]) for q in Q1 for m in ("jev14", "drex")))))
    return out


# ================================================================ 9  printing, writing, checking
def flat(o, pre=""):
    """Every number of a nested result, with its path."""
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, pre + "/" + str(k))
    elif isinstance(o, (list, tuple)):
        for k, v in enumerate(o): yield from flat(v, pre + "/" + str(k))
    elif isinstance(o, bool) or o is None or isinstance(o, str): yield pre, o
    else: yield pre, float(o)


def main():
    opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d; B = int(opt("--boot", 1000)); out_dir = HERE / opt("--out", "results"); out_dir.mkdir(exist_ok=True)
    Q, P, S1, F, D, calls = load(); name = {q["id"]: q["name"] for q in Q}; S1["allq"] = Q; S1["sign"] = {q["id"]: q["sign"] for q in Q}; S1["name"] = name; f2 = lambda v: "%.2f" % v; iv = lambda c: "[%.2f, %.2f]" % tuple(c)
    print("data: first study %d raters, %d ratings of %d excerpts | second study %d raters, %d ratings of %d excerpts | pool %d passages of %d recordings" % (
        S1["raters"], len(S1["rows"]), len(S1["ex"]), D["n_valid"], int(np.isfinite(D["y"]).sum()), len(F["ex"]), len(P["cand"]), len(set(P["rid"]))))

    print("\n=== FIRST STUDY, fixed in advance: twelve questions (rebuilt for the corpus; verdict of the random part alone beside it)")
    a1, tog = study1_fixed(S1, B)
    print("%-17s %-4s %-9s %-19s %-19s %-19s %s" % ("question", "dim", "one rater", "mean of five", "jev-1.14.0", "drex-max", "verdict (random part alone)"))
    for q in sorted(S1["Q"], key=lambda q: -a1[q]["rebuilt"]["reliability_of_mean"]):
        r = a1[q]["rebuilt"]; print("%-17s %-4s %-9s %-19s %-19s %-19s %s (%s)" % (name[q], S1["dimension"][q], f2(r["reliability_one_rater"]), f2(r["reliability_of_mean"]) + " " + iv(r["reliability_of_mean_interval"]), f2(r["jev14"]["correlation"]) + " " + iv(r["jev14"]["interval"]),
              f2(r["drex"]["correlation"]) + " " + iv(r["drex"]["interval"]), a1[q]["verdict"], a1[q]["verdict_random_part_alone"]))
    print("the two models over the twelve questions: jev-1.14.0 %.3f, drex-max %.3f, difference %+.3f [%+.3f, %+.3f]" % (tog["mean_correlation_jev14"], tog["mean_correlation_drex"], tog["mean_difference_of_correlations"], *tog["mean_difference_of_correlations_interval"]))
    print("\n=== FIRST STUDY, added afterwards")
    f1 = study1_further(S1, P, B); t = f1["together"]
    print("against an unlimited panel: jev-1.14.0 %.2f %s, drex-max %.2f %s, one rater %.2f %s" % (t["with_panel"]["jev14"]["value"], iv(t["with_panel"]["jev14"]["interval"]), t["with_panel"]["drex"]["value"], iv(t["with_panel"]["drex"]["interval"]), t["with_panel_one_rater"]["value"], iv(t["with_panel_one_rater"]["interval"])))
    print("dimensions from their marker questions: " + " ; ".join("%s %s (%s): %.2f, models %.2f / %.2f" % (f, v["by_the_rule_for_questions"], "+".join(v["names"]), v["reliability_of_mean"], v["dimension_jev14"]["with_mean"], v["dimension_drex"]["with_mean"]) for f, v in f1["dimensions"].items()))
    g = f1["known_groups"]["random_part"]["across_questions"]; print("known groups, random part: streams minus videos, people against jev-1.14.0 r = %.2f %s, against drex-max %.2f; per year %.2f, %.2f" % (g["kind_jev14"]["value"], iv(g["kind_jev14"]["interval"]), g["kind_drex"]["value"], g["year_jev14"]["value"], g["year_drex"]["value"]))
    lv = f1["questions"]; print("levels, model minus people (scale points): largest for jev-1.14.0: %s %+.2f; drex-max lower than people on %d of 12 questions" % (max(((name[q], v["levels"]["jev14"]["minus_people"]) for q, v in lv.items()), key=lambda x: abs(x[1])) + (sum(v["levels"]["drex"]["minus_people"] < 0 for v in lv.values()),)))

    print("\n=== SECOND STUDY, fixed in advance: eight dimensions (plain figures, stated for five raters per question)")
    r2, ctx = study2_fixed(F, D, a1, B)
    print("%-3s %-45s %-19s %-19s %-19s %-15s %s" % ("", "dimension", "reliability", "jev-1.14.0", "drex-max", "verdict", "settled?"))
    for f, d in r2["dimensions"].items():
        p = d["plain"]; print("%-3s %-45s %-19s %-19s %-19s %-15s %s" % (f, d["label"], f2(p["reliability"]) + " " + iv(p["reliability_interval"]), f2(p["jev14_with_mean"]) + " " + iv(p["jev14_interval"]), f2(p["drex_with_mean"]) + " " + iv(p["drex_interval"]), p["verdict"],
              "settled" if p["settled"] else "at the line (%s)" % ", ".join("%s %.0f%%" % (v, 100 * s) for v, s in list(p["verdict_in_resamples"].items())[:2])))
    print("with the model's help | recordings")
    for f, d in r2["dimensions"].items():
        w, rc = d["with_the_model"], d["recordings"]; print("%-3s reliability %s %s, panel: jev %s %s, drex %s %s, %-14s | share between recordings %s %s, site score with a panel %s %s" % (f, f2(w["reliability"]), iv(w["reliability_interval"]), f2(w["jev14_with_panel"]), iv(w["jev14_with_panel_interval"]),
              f2(w["drex_with_panel"]), iv(w["drex_with_panel_interval"]), w["verdict"], f2(rc["share_between_recordings"]), iv(rc["share_interval"]), f2(rc["site_score_with_panel"]), iv(rc["site_score_interval"])))
    o8 = r2["models"]["over_the_eight"]; du = r2["dimensions_usable"]; print("usable: %d by the plain figures (settled %d), %d with the model's help | the two models over the eight: %.3f and %.3f, difference %+.3f [%+.3f, %+.3f]" % (du["plain"], du["settled_as_usable"], du["with_the_model"], o8["jev14"], o8["drex"], o8["difference"], *o8["difference_interval"]))
    print("the sixteen questions (mean of five | jev-1.14.0 | drex-max | by the rule): " + " ; ".join("%s %.2f | %.2f | %.2f | %s" % (v["name"], v["reliability"], v["jev14_with_mean"], v["drex_with_mean"], v["verdict_not_used"]) for v in r2["questions"].values()))
    ar = r2["arms"]["raters"]; print("the two arms (sample answers | own answers only): correlation with jev-1.14.0 %.3f | %.3f, difference %+.3f [%+.3f, %+.3f]; with the other raters %.3f | %.3f; spread %.2f | %.2f" % (ar["with_jev14"]["first"], ar["with_jev14"]["second"], ar["with_jev14"]["difference"], *ar["with_jev14"]["interval"],
          ar["with_others"]["first"], ar["with_others"]["second"], ar["spread"]["first"], ar["spread"]["second"]))
    rp = repeat_calls(calls, Q, F["fids"]); print("four calls of the same text: reliability of one call, middle question jev-1.14.0 %.3f, drex-max %.3f; dimensions %.3f to %.3f and %.3f to %.3f" % (np.median([v["reliability_of_one_call"] for v in rp["jev14"]["questions"].values()]),
          np.median([v["reliability_of_one_call"] for v in rp["drex"]["questions"].values()]), min(v["reliability_of_one_call"] for v in rp["jev14"]["dimensions"].values()), max(v["reliability_of_one_call"] for v in rp["jev14"]["dimensions"].values()),
          min(v["reliability_of_one_call"] for v in rp["drex"]["dimensions"].values()), max(v["reliability_of_one_call"] for v in rp["drex"]["dimensions"].values())))

    print("\n=== SECOND STUDY, added afterwards (first round)")
    fu = study2_further(F, D, r2, ctx, f1["dimensions"], B); a = fu["replication"]["across_the_twelve"]
    print("the first study repeated, twelve questions: models' correlations r = %.2f (jev-1.14.0), %.2f (drex-max); reliabilities r = %.2f, means %.2f and %.2f (levels left in %.2f)" % (a["jev14"]["correlation"], a["drex"]["correlation"], a["reliability"]["correlation"], a["reliability"]["mean_first"], a["reliability"]["mean_second"], a["reliability"]["mean_second_levels_in"]))
    print("a rater's level: %.0f%% to %.0f%% of the differences between single ratings | dimension score against its own composite: %.2f, %.2f; against the others %.2f, %.2f" % (100 * min(v["share_level"] for v in fu["sources"].values()), 100 * max(v["share_level"] for v in fu["sources"].values()),
          fu["cross"]["jev14"]["mean_own"], fu["cross"]["drex"]["mean_own"], fu["cross"]["jev14"]["mean_other"], fu["cross"]["drex"]["mean_other"]))
    ps = fu["panel_summary"]; gq = fu["groups"]["across_questions"]["streams_minus_videos"]["the_12_rated_before"]; print("against an unlimited panel: jev-1.14.0 %.2f, drex-max %.2f, one rating of each marker %.2f | streams minus videos, people against jev-1.14.0 across the twelve questions r = %.2f" % (ps["jev14"]["mean"], ps["drex"]["mean"], ps["one_rating_of_each_marker"]["mean"], gq["jev14"]["correlation"]))
    print("\n=== SECOND STUDY, added afterwards (second round)")
    mo = study2_more(F, D, r2, fu, ctx, a1, f1, B)
    print("with raters as well as recordings: " + " ; ".join("%s reliability %s (%s)" % (f, iv(v["reliability"]["interval_both"]), "settled" if v["settled_both"] else "at the line") for f, v in mo["raters"]["dimensions"].items()))
    print("five rounds as panels, no model (levels left in) against the model's figure: " + " ; ".join("%s %.2f | %.2f" % (f, v["five_raters_per_question"], v["model_levels_in"]) for f, v in mo["rounds"]["dimensions"].items()))
    print("pooled, usable and settled: %s" % ", ".join(mo["pooled"]["summary"]["usable_and_settled"])); wsum = mo["within"]["summary"]
    print("within a recording against between recordings, against the panel: jev-1.14.0 %.2f | %.2f, drex-max %.2f | %.2f" % (wsum["jev14"]["mean_within_with_panel"], wsum["jev14"]["mean_between_with_panel"], wsum["drex"]["mean_within_with_panel"], wsum["drex"]["mean_between_with_panel"]))
    print("at the same model score, streams against videos (sd): " + " ; ".join("%s %+.2f %s" % (f, v["jev14"]["streams"], "[%+.2f, %+.2f]" % tuple(v["jev14"]["streams_interval"])) for f, v in mo["bias"]["dimensions"].items()))

    print("\n=== A THIRD MODEL, d1.2-ultra, on both studies (added afterwards)")
    th = third_model(S1, F, D, ctx, a1, rp, B)
    for label, o in (("second study, eight dimensions", th["second_study"]["over_the_eight_dimensions"]), ("second study, sixteen questions", th["second_study"]["over_the_sixteen_questions"]), ("first study, twelve questions", th["first_study"]["over_the_twelve_questions"])):
        print("%-32s jev-1.14.0 %.3f, drex-max %.3f, d1.2-ultra %.3f | d1.2-ultra minus jev-1.14.0 %+.3f [%+.3f, %+.3f], minus drex-max %+.3f [%+.3f, %+.3f]" % (label, o["jev14"]["mean"], o["drex"]["mean"], o["d12u"]["mean"], o["d12u_minus_jev14"]["difference"], *o["d12u_minus_jev14"]["interval"], o["d12u_minus_drex"]["difference"], *o["d12u_minus_drex"]["interval"]))
    ls = th["second_study"]["levels_summary"]; print("mean distance of a model's level from people's (scale points): jev-1.14.0 %.2f, drex-max %.2f, d1.2-ultra %.2f" % (ls["jev14"]["mean_absolute_difference"], ls["drex"]["mean_absolute_difference"], ls["d12u"]["mean_absolute_difference"]))
    results = dict(study1_fixed=a1, study1_together=tog, study1_further=f1, study2_fixed=r2, study2_repeat_calls=rp, study2_further=fu, study2_more=mo, third_model=th)
    for k, v in results.items(): (out_dir / (k + ".json")).write_text(json.dumps(v, indent=1, default=float) + "\n", encoding="utf-8")
    print("\nwrote %d files to %s" % (len(results), out_dir))
    if "--write-expected" in sys.argv:
        (HERE / "expected.json").write_text(json.dumps({k: v for k, v in flat(results)}, indent=0) + "\n", encoding="utf-8"); print("wrote expected.json")
    if "--check" in sys.argv:
        exp = json.loads((HERE / "expected.json").read_text(encoding="utf-8")); got = dict(flat(results)); bad = []
        for k, v in exp.items():
            g = got.get(k, "missing")
            same = (isinstance(v, float) and isinstance(g, float) and ((math.isnan(v) and math.isnan(g)) or abs(v - g) <= 1e-6 + 1e-6 * abs(v))) or v == g
            if not same: bad.append((k, v, g))
        print("check against expected.json: %d values compared, %d differ%s" % (len(exp), len(bad), "" if B == 1000 else " (intervals differ when --boot is not 1000)"))
        for k, v, g in bad[:20]: print("   %s: expected %s, got %s" % (k, v, g))
        sys.exit(1 if bad and B == 1000 else 0)


if __name__ == "__main__": main()
