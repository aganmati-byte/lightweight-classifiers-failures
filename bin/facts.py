import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'results' / '2026-11-11_paper_assets'
from scipy import stats
P = ASSETS / 'data'
S, M, K, C, D, DS = [pd.read_csv(P / f'{n}.csv') for n in ('S', 'M', 'K', 'C', 'D', 'DS')]
meta = json.loads((P / 'meta.json').read_text())
F = {}
def dl(est, se):
    est, se = np.asarray(est, float), np.asarray(se, float); k = len(est); w = 1 / se**2
    fe = (w * est).sum() / w.sum(); Q = (w * (est - fe)**2).sum()
    tau2 = max(0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum())) if k > 1 else 0
    wr = 1 / (se**2 + tau2); mu = (wr * est).sum() / wr.sum(); s = np.sqrt(1 / wr.sum()); t = stats.t.ppf(.975, max(1, k - 2))
    return dict(k=k, pooled=mu, lo=mu - 1.96 * s, hi=mu + 1.96 * s, pi_lo=mu - t * np.sqrt(tau2 + s**2), pi_hi=mu + t * np.sqrt(tau2 + s**2),
                tau2=tau2, I2=max(0, (Q - (k - 1)) / Q) if Q > 0 else 0, Q=Q, Q_p=1 - stats.chi2.cdf(Q, k - 1) if k > 1 else np.nan)
def se_of(e, col):
    return (e[col + '_ci_high'] - e[col + '_ci_low']) / (2 * stats.t.ppf(.975, e.splits - 1))
# --- meta-analyses
for b in (0.1, 0.2):
    for cand in ['rf_thinned', 'rf_direct_small', 'gbm_truncated', 'gbm_direct_small']:
        e = S[(S.budget == b) & (S.candidate == cand)].copy()
        for col in ['excess_newly_missed_per100', 'excess_net_tp_lost_per100', 'excess_delta_mcc']:
            r = dl(e[col], se_of(e, col)); npos = int((e[col] > 0).sum())
            r.update(npos=npos, sign_p=stats.binomtest(npos, len(e), .5, alternative='greater').pvalue)
            if col == 'excess_newly_missed_per100':
                loo = [dl(e[e.dataset != d][col], se_of(e[e.dataset != d], col))['pooled'] for d in e.dataset]
                r.update(loo_min=min(loo), loo_max=max(loo))
            F[f'meta|{b}|{cand}|{col}'] = r
# --- per-dataset
for _, r in S.iterrows():
    F[f'ds|{r.budget}|{r.candidate}|{r.dataset}'] = {k: r[k] for k in ['excess_newly_missed_per100', 'excess_newly_missed_per100_ci_low', 'excess_newly_missed_per100_ci_high',
        'excess_net_tp_lost_per100', 'excess_delta_mcc', 'excess_delta_mcc_ci_low', 'excess_delta_mcc_ci_high', 'share_splits_excess_missed_gt0', 'splits',
        'newly_missed', 'retrain_noise_newly_missed']}
# --- accuracy vs MCC
Mb = M[(M.budget == 0.1) & ~M.model.str.contains('retrain')].copy()
Mb['accuracy'] = (Mb.TP + Mb.TN) / (Mb.TP + Mb.TN + Mb.FP + Mb.FN)
med = Mb.groupby(['dataset', 'model'])[['accuracy', 'mcc', 'sensitivity']].median()
acc = {}
for ds, g in med.groupby(level=0):
    g = g.droplevel(0)
    acc[ds] = dict(acc_range=g.accuracy.max() - g.accuracy.min(), mcc_range=g.mcc.max() - g.mcc.min(), sens_range=g.sensitivity.max() - g.sensitivity.min(),
                   best_acc=g.accuracy.idxmax(), best_mcc=g.mcc.idxmax(), acc_min=g.accuracy.min(), acc_max=g.accuracy.max(), mcc_min=g.mcc.min(), mcc_max=g.mcc.max())
F['accmcc'] = acc
# --- noise floor (retrained vs reference, per 100 events)
Cn = C[(C.budget == 0.1) & (C.candidate == 'rf_thinned')]
F['noise_rf_per100'] = (100 * Cn.newly_missed_retrain_mean / Cn.events_test).groupby(Cn.dataset).mean().to_dict()
F['thin_rf_per100'] = (100 * Cn.newly_missed_candidate / Cn.events_test).groupby(Cn.dataset).mean().to_dict()
# --- mechanism
Kb = K[K.budget == 0.1]
F['mech'] = {ds: dict(exp=g.expected_newly_missed.sum(), obs=g.observed_newly_missed.sum(), auroc=g.auroc_predicting_changes.mean(),
                       auroc_min=g.auroc_predicting_changes.min(), splits=len(g)) for ds, g in Kb.groupby('dataset')}
F['mech_all'] = dict(exp=Kb.expected_newly_missed.sum(), obs=Kb.observed_newly_missed.sum(),
                     r=np.corrcoef(Kb.expected_newly_missed, Kb.observed_newly_missed)[0, 1], n=len(Kb),
                     auroc_median=Kb.auroc_predicting_changes.median(), auroc_min_ds=min(v['auroc'] for v in F['mech'].values()),
                     auroc_max_ds=max(v['auroc'] for v in F['mech'].values()))
parse = lambda c: np.vstack(Kb[c].apply(lambda v: json.loads(v) if isinstance(v, str) else v))
nq = parse('margin_quartile_n').sum(0)
F['quartiles'] = dict(thinned=(100 * parse('margin_quartile_lost_thinned').sum(0) / nq).tolist(),
                      retrained=(100 * parse('margin_quartile_lost_retrain_mean').sum(0) / nq).tolist(), n=nq.tolist())
# --- dose
Db = D[D.budget == 0.1]
F['dose'] = {f'{r.dataset}|{int(r.trees)}': dict(ex=r.excess_missed_per100, lo=r.ci_low, hi=r.ci_high, exp=r.expected_missed, obs=r.observed_missed,
                                                 auroc=r.auroc_flip, dmcc=r.excess_delta_mcc) for _, r in Db.iterrows()}
rho = stats.spearmanr(-Db.trees, Db.excess_missed_per100)
F['dose_trend'] = dict(spearman_rho=rho.statistic, p=rho.pvalue, n=len(Db),
                       monotone=[ds for ds, g in Db.groupby('dataset') if np.all(np.diff(g.sort_values('trees', ascending=False).excess_missed_per100) > -0.5)])
F['dose_mech'] = dict(r=np.corrcoef(Db.expected_missed, Db.observed_missed)[0, 1], exp=Db.expected_missed.sum(), obs=Db.observed_missed.sum())
F['sizes'] = DS.groupby(['dataset', 'trees']).bytes.first().to_dict()
F['meta_data'] = {k: {kk: v.get(kk) for kk in ('n', 'events', 'prevalence', 'n_groups')} for k, v in meta.items()}
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
(ASSETS / 'facts.json').write_text(json.dumps(clean(F), indent=1))
# ---------- print key numbers
for b in (0.1, 0.2):
    for cand in ['rf_thinned', 'rf_direct_small', 'gbm_truncated', 'gbm_direct_small']:
        r = F[f'meta|{b}|{cand}|excess_newly_missed_per100']; m = F[f'meta|{b}|{cand}|excess_delta_mcc']; n = F[f'meta|{b}|{cand}|excess_net_tp_lost_per100']
        print(f"{b} {cand:17s} missed {r['pooled']:+.2f} [{r['lo']:+.2f},{r['hi']:+.2f}] PI[{r['pi_lo']:+.2f},{r['pi_hi']:+.2f}] I2={100*r['I2']:.0f}% pos {r['npos']}/{r['k']} p={r['sign_p']:.4f} LOO[{r['loo_min']:+.2f},{r['loo_max']:+.2f}] | net {n['pooled']:+.2f} [{n['lo']:+.2f},{n['hi']:+.2f}] | dMCC {m['pooled']:+.4f} [{m['lo']:+.4f},{m['hi']:+.4f}] pos {m['npos']}")
print(pd.DataFrame(acc).T[['acc_range', 'mcc_range', 'sens_range', 'best_acc', 'best_mcc']].round(3))
print('noise', {k: round(v, 1) for k, v in F['noise_rf_per100'].items()})
print('thin ', {k: round(v, 1) for k, v in F['thin_rf_per100'].items()})
print('mech_all', {k: round(v, 3) for k, v in F['mech_all'].items()})
print('quartiles', {k: [round(x, 1) for x in v] for k, v in F['quartiles'].items()})
print('dose_trend', F['dose_trend']); print('dose_mech', F['dose_mech'])
print(Db[['dataset', 'trees', 'excess_missed_per100', 'ci_low', 'ci_high', 'expected_missed', 'observed_missed']].round(2).to_string())
