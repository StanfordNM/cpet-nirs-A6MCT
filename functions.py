import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu
from collections import defaultdict
from itertools import combinations
import os
import re

plt.rcParams['font.family'] = 'Times New Roman'

DIAGNOSIS_COLORS = {'Control': '#808080',
                    'SMA': '#457B9D',
                    'DMD': '#E9C46A'}

# -------------------------------

def median_range(s):
    s = s.dropna()
    return 'N/A' if s.empty else f"{s.median():.1f} [{s.min():.1f}, {s.max():.1f}]"

def median_iqr(s):
    s = s.dropna()
    return 'N/A' if s.empty else f"{s.median():.1f} [{s.quantile(0.25):.1f}, {s.quantile(0.75):.1f}]"

def pval_str(p):
    return 'N/A' if np.isnan(p) else '<0.001' if p < 0.001 else f"{p:.3f}"

def bh_adjust(ps):
    ps = np.asarray(ps, dtype=float)
    adj = ps.copy()
    ok = ~np.isnan(ps)
    if ok.any():
        adj[ok] = stats.false_discovery_control(ps[ok], method='bh')
    return adj

# -------------------------------

def attach_nearest_test(demo, tests, value_cols, only_dx=None): # for PUL and RULM
    new_cols = value_cols + [f'{c}_gap_days' for c in value_cols]
    demo = demo.drop(columns=[c for c in demo.columns if c in new_cols])
    t = tests[['participant_id'] + new_cols]
    if only_dx is not None:
        t = t[t['participant_id'].isin(demo.loc[demo['clinical_diagnosis'] == only_dx, 'participant_id'])]
    return demo.merge(t, on='participant_id', how='left', validate='1:1')

# -------------------------------
#### DEMOGRAPHICS
# -------------------------------

def get_patient_info(cpet_demographics):
    results_df = cpet_demographics.drop_duplicates('participant_id')

    def sex_pct(s):
        s = s.dropna()
        if s.empty:
            return 'N/A'
        male_count = (s.str.lower() == 'male').sum()
        female_count = (s.str.lower() == 'female').sum()
        total = male_count + female_count
        if total == 0:
            return 'N/A'
        return f"M: {male_count} ({male_count/total*100:.0f}%) / F: {female_count} ({female_count/total*100:.0f}%)"

    diag_counts = results_df.groupby('clinical_diagnosis')['participant_id'].count()
    col_labels = {diag: f"{diag} (N = {diag_counts[diag]})" for diag in diag_counts.index}
    group = results_df.groupby('clinical_diagnosis')

    rows = {
        'Age at visit (y)': group['age_at_cpet'].apply(median_range),
        'Sex (%), M/F': group['sex'].apply(sex_pct),
        'Height (m)': group['height'].apply(median_range),
        'Weight (kg)': group['weight'].apply(median_range),
        'BMI (kg/m²)': group['bmi'].apply(median_range),
        'PUL': group['pul_total'].apply(median_range),
        'RULM': group['rulm_best'].apply(median_range)
    }

    summary = pd.DataFrame(rows).T
    summary = summary[['Control', 'SMA', 'DMD']].rename(columns=col_labels)
    groups = ['Control', 'SMA', 'DMD']

    def continuous_test(col):
        v = {g: results_df.loc[results_df['clinical_diagnosis'] == g, col].dropna() for g in groups}
        ok = [g for g in groups if len(v[g]) >= 2]
        return stats.kruskal(*[v[g] for g in ok])[1] if len(ok) >= 2 else np.nan

    def sex_test():
        table = np.array([((results_df.loc[results_df['clinical_diagnosis'] == g, 'sex'] == 'Male').sum(),
                           (results_df.loc[results_df['clinical_diagnosis'] == g, 'sex'] == 'Female').sum()) for g in groups])
        return stats.chi2_contingency(table)[1] if (table.sum(axis=0) > 0).all() else np.nan

    tests = {'Age at visit (y)': continuous_test('age_at_cpet'), 'Sex (%), M/F': sex_test(),
             'Height (m)': continuous_test('height'), 'Weight (kg)': continuous_test('weight'),
             'BMI (kg/m²)': continuous_test('bmi')}
    summary['P-value'] = [pval_str(tests[r]) if r in tests else '' for r in summary.index]
    return summary

# -------------------------------
#### CPET FUNCTIONS
# -------------------------------

def apply_30s_rolling_average(df, window_sec=30):
    result = df.copy()
    columns = [col for col in df.columns if col not in ['t', 'Phase']]
    t = df['t'].values
    vals = df[columns].values
    half_window = window_sec / 2
    lo_idx = np.searchsorted(t, t - half_window, side='left')
    hi_idx = np.searchsorted(t, t + half_window, side='right')
    smoothed = np.array([np.nanmean(vals[lo:hi], axis=0) for lo, hi in zip(lo_idx, hi_idx)])
    result[columns] = smoothed
    return result

# -------------------------------

def _find_sustained_crossing(sub_df, col, half_val, min_sustain_sec):
    t = sub_df['t'].values
    v = sub_df[col].values
    below = v <= half_val
    for i in range(len(t)):
        if not below[i]:
            continue
        window_mask = (t >= t[i]) & (t <= t[i] + min_sustain_sec)
        if below[window_mask].all():
            return t[i]
    return None

# -------------------------------

def compute_recovery_metrics(df, min_sustain_sec=0):
    
    if 'VO2/HR' not in df.columns:
        if 'HR' in df.columns:
            df = df.copy()
            df['VO2/HR'] = df['VO2'] / df['HR']

    df = apply_30s_rolling_average(df)
    exercise_end_time = df[df['Phase'] == 'EXERCISE']['t'].max()
    df.loc[df['t'] > exercise_end_time, 'Phase'] = 'RECOVERY' # sometimes it's REST for a portion of the recovery period

    anchor_idx = df[df['Phase'] == 'EXERCISE']['t'].idxmax()
    anchor_t = df.loc[anchor_idx, 't']

    t1_2_cols = ['VO2', 'VE', 'VCO2'] + (['VO2/HR'] if 'VO2/HR' in df.columns else []) + (['HR'] if 'HR' in df.columns else [])
    overshoot_cols = ['RQ', 'VE/VO2', 'VE/VCO2', 'PetO2']

    metrics = {}
    for col in t1_2_cols:
        anchor_val = df.loc[anchor_idx, col]
        baseline_val = df.loc[df['Phase'] == 'REST', col].median()
        metrics[f'{col}_t1_2_no_decline'] = anchor_val <= baseline_val
        if anchor_val <= baseline_val:
            metrics[f'{col}_t1_2'] = np.nan
            continue
        half_val = baseline_val + (anchor_val - baseline_val) / 2
        post_anchor = df[df['t'] > anchor_t]
        crossing_t = _find_sustained_crossing(post_anchor, col, half_val, min_sustain_sec)
        metrics[f'{col}_t1_2'] = np.nan if crossing_t is None else crossing_t - anchor_t

    for col in overshoot_cols:
        anchor_val = df.loc[anchor_idx, col]
        window = df[(df['t'] >= anchor_t) & (df['t'] <= anchor_t + 300)]
        if window.empty or anchor_val == 0:
            metrics[f'{col}_overshoot_pct'] = np.nan
        else:
            metrics[f'{col}_overshoot_pct'] = ((window[col].max() - anchor_val) / abs(anchor_val)) * 100

    return metrics

# -------------------------------

def get_first_scan_per_patient(all_cpet_files):
    files = {os.path.basename(fp).split('_')[0]: fp for fp in all_cpet_files}
    assert len(files) == len(all_cpet_files), 'expected one CPET file per participant'
    return files

# -------------------------------

def _pairwise_fdr(group_data, metric_cols, pairs, binary_cols=()):
    adjusted_pvals = {}
    for g1, g2 in pairs:
        ps = []
        for col in metric_cols:
            a1 = group_data[g1][col].dropna().values
            a2 = group_data[g2][col].dropna().values
            if len(a1) < 2 or len(a2) < 2:
                ps.append(np.nan)
            elif col in binary_cols:
                ct = np.array([[int((a1 == 1).sum()), int((a1 == 0).sum())],
                               [int((a2 == 1).sum()), int((a2 == 0).sum())]])
                ps.append(stats.fisher_exact(ct)[1])
            else:
                ps.append(stats.mannwhitneyu(a1, a2, alternative='two-sided')[1])
        adjusted_pvals[f"{g1} vs. {g2}"] = dict(zip(metric_cols, bh_adjust(ps)))
    return adjusted_pvals

# -------------------------------

def make_recovery_table(all_cpet_files, cpet_demographics, min_sustain_sec=0):
    rows = []
    for participant_id, fp in get_first_scan_per_patient(all_cpet_files).items():
        demo = cpet_demographics[cpet_demographics['participant_id'] == participant_id]
        if demo.empty:
            continue
        df = pd.read_csv(fp).sort_values('t').reset_index(drop=True)
        m = compute_recovery_metrics(df, min_sustain_sec=min_sustain_sec)
        m['participant_id'] = participant_id
        m['clinical_diagnosis'] = demo['clinical_diagnosis'].values[0]
        rows.append(m)

    data = pd.DataFrame(rows)
    groups = ['Control', 'SMA', 'DMD']
    group_data = {g: data[data['clinical_diagnosis'] == g] for g in groups}

    metric_labels = {
        'VO2_t1_2': 'T1/2 VO₂ (s)',
        'VE_t1_2': 'T1/2 VE (s)',
        'VCO2_t1_2': 'T1/2 VCO₂ (s)',
        'VO2/HR_t1_2': 'T1/2 O₂ pulse (s)',
        'RQ_overshoot_pct': 'Overshoot of RQ (%)',
        'VE/VO2_overshoot_pct': 'Overshoot of VE/VO₂ (%)',
        'VE/VCO2_overshoot_pct': 'Overshoot of VE/VCO₂ (%)',
        'PetO2_overshoot_pct': 'Overshoot of PETO₂ (%)',
    }

    no_decline = data.groupby('clinical_diagnosis')[[c for c in data.columns if c.endswith('_no_decline')]].sum().reindex(groups)
    print('T1/2 set to NaN (end-exercise value <= resting baseline), n per group:')
    print(no_decline.fillna(0).astype(int).T)

    pairs = [('Control', 'SMA'), ('Control', 'DMD'), ('SMA', 'DMD')]
    adjusted_pvals = _pairwise_fdr(group_data, list(metric_labels), pairs)

    results = []
    for col, label in metric_labels.items():
        row = {'Metric': label}
        for g in groups:
            row[f"{g} (N={len(group_data[g])})"] = median_iqr(group_data[g][col])
        for key, ps in adjusted_pvals.items():
            row[key] = pval_str(ps[col])
        results.append(row)

    return pd.DataFrame(results).set_index('Metric')

# -------------------------------

def predict_vo2_arm_p50(age, sex, height_m, weight_kg): # Shakespeare & Parkes 2025
    """Age in years, sex (0 = male, 1 = female), height in metres, weight in kg"""
    s = 1 if str(sex).strip().upper() in ('F', 'FEMALE') else 0 if str(sex).strip().upper() in ('M', 'MALE') else np.nan
    if any(pd.isna(x) for x in (age, s, height_m, weight_kg)):
        return np.nan
    return 853.0593 - 12.27326*age - 658.9721*s + 761.8335*height_m + 6.190653*weight_kg

def extract_peak_cpet_metrics(fp, age, sex, height_m, weight_kg, rest_window_s=60):
    raw = pd.read_csv(fp).sort_values('t').reset_index(drop=True)
    rest_vo2_mL = raw.loc[(raw['Phase'] == 'REST') & (raw['t'] < 0) & (raw['t'] >= -rest_window_s), 'VO2'].mean()
    df = apply_30s_rolling_average(raw)
    exercise = df[df['Phase'] == 'EXERCISE']
    if exercise.empty:
        return {}

    if 'VO2/HR' not in exercise.columns:
        if 'HR' in exercise.columns:
            exercise = exercise.copy()
            exercise['VO2/HR'] = exercise['VO2'] / exercise['HR']

    pred_vo2 = predict_vo2_arm_p50(age, sex, height_m=height_m, weight_kg=weight_kg)
    pred_hr = 208 - (0.7 * age)
    peak_vo2_mL = exercise['VO2'].max()
    net_peak_vo2_mL = peak_vo2_mL - rest_vo2_mL
    peak_vo2_mLkg = exercise['VO2/Kg'].max()
    peak_rer = exercise['RQ'].max()
    if peak_rer > 2.0:
        print(f"Warning: {os.path.basename(fp)} — peak RER exceeds 2.0 ({peak_rer})")

    rer_above_1 = exercise[exercise['RQ'] >= 1.0]
    sustained_max_rer = (not rer_above_1.empty and not rer_above_1.groupby((rer_above_1['t'].diff() > 1).cumsum()).filter(lambda x: len(x) >= 60).empty)

    peak_hr = exercise['HR'].max() if 'HR' in exercise.columns else np.nan
    peak_o2_pulse = exercise['VO2/HR'].max() if 'VO2/HR' in exercise.columns else np.nan

    return {
        'VO2_rest_L': rest_vo2_mL / 1000,
        'VO2_peak_L': peak_vo2_mL / 1000,
        'net_peak_vo2_above_rest_L': net_peak_vo2_mL / 1000,
        'VO2_peak_mLkgmin': peak_vo2_mLkg,
        'pred_vo2_pct': (peak_vo2_mL / pred_vo2 * 100) if pred_vo2 > 0 else np.nan,
        'peak_RER': peak_rer,
        'sustained_max_RER': sustained_max_rer,
        'peak_HR': peak_hr,
        'pred_HR_pct': (peak_hr / pred_hr * 100) if (not np.isnan(peak_hr) and pred_hr > 0) else np.nan,
        'O2_pulse': peak_o2_pulse,
    }

# -------------------------------

def make_peak_cpet_table(all_cpet_files, cpet_demographics, rev_df, return_data=False):

    rev_df = rev_df.dropna(subset=['mct_cum6'])
    rows = []
    for participant_id, fp in get_first_scan_per_patient(all_cpet_files).items():
        demo = cpet_demographics[cpet_demographics['participant_id'] == participant_id]
        if demo.empty:
            continue
        demo_row = demo.iloc[0]
        m = extract_peak_cpet_metrics(fp, demo_row['age_at_cpet'], demo_row['sex'], demo_row['height'], demo_row['weight'])
        if not m:
            continue
        m['participant_id'] = participant_id
        m['clinical_diagnosis'] = demo_row['clinical_diagnosis']
        rev = rev_df[rev_df['participant_id'] == participant_id]
        m['mct_cum6'] = np.nan if rev.empty else rev.iloc[0]['mct_cum6']
        rows.append(m)

    data = pd.DataFrame(rows)
    data['reached_max_RER'] = data['sustained_max_RER'].astype(int)
    groups = ['Control', 'SMA', 'DMD']
    group_data = {g: data[data['clinical_diagnosis'] == g] for g in groups}

    metric_labels = {
        'VO2_rest_L': 'Resting VO₂ (L/min)',
        'VO2_peak_L': 'Peak VO₂ (L/min)',
        'net_peak_vo2_above_rest_L': 'Net peak VO₂ above rest (L/min)',
        'VO2_peak_mLkgmin': 'Peak VO₂ (mL/kg/min)',
        'pred_vo2_pct': 'Predicted Peak VO₂ (%)',
        'peak_RER': 'Peak RER (no unit)',
        'reached_max_RER': 'Peak RER ≥ 1.0 for ≥ 60s, n (%)',
        'peak_HR': 'Peak Heart Rate (bpm)',
        'pred_HR_pct': 'Predicted Heart Rate (%)',
        'O2_pulse': 'O₂ pulse (mL/bpm)',
        'mct_cum6': 'Cumulative revolutions',
    }
    pairs = [('Control', 'SMA'), ('Control', 'DMD'), ('SMA', 'DMD')]
    adjusted_pvals = _pairwise_fdr(group_data, list(metric_labels), pairs, binary_cols=('reached_max_RER',))

    results = []
    for col, label in metric_labels.items():
        row = {'Metric': label}
        for g in groups:
            if col == 'reached_max_RER':
                n_total = len(group_data[g])
                n_max = int(group_data[g]['reached_max_RER'].sum())
                row[f"{g} (N={n_total})"] = f"{n_max} ({n_max/n_total*100:.0f}%)" if n_total > 0 else 'N/A'
            else:
                row[f"{g} (N={len(group_data[g])})"] = median_iqr(group_data[g][col])
        for key, ps in adjusted_pvals.items():
            row[key] = pval_str(ps[col])
        results.append(row)

    table = pd.DataFrame(results).set_index('Metric')
    return (table, data) if return_data else table

# -------------------------------
#### NIRS FUNCTIONS!
# -------------------------------

def build_supp_table3(df):

    GROUPS = ['Control', 'SMA', 'DMD']
    MUSCLES = {'Triceps': dict(optode_cols=['tri_optode_1', 'tri_optode_2', 'tri_optode_3'], caliper='triceps_caliper'),
               'Biceps': dict(optode_cols=['bi_optode_1', 'bi_optode_2', 'bi_optode_3'], caliper='biceps_caliper')}
    DEPTH_LABELS = {1: 'Optode 1 (1.5cm depth)', 2: 'Optode 2 (1.75cm depth)', 3: 'Optode 3 (2.25cm depth)', 0: 'No acceptable signal'}

    def get_deepest_optodes(df, optode_cols):
        # highest-numbered usable optode per participant, 0 if none
        assert df[optode_cols].notna().all().all()
        flags = df[optode_cols].to_numpy(dtype=bool)
        k = len(optode_cols)
        return pd.Series(np.where(flags.any(axis=1), k - np.argmax(flags[:, ::-1], axis=1), 0), index=df.index)

    df = df[df['diagnosis'].isin(GROUPS)].copy()
    n = df['diagnosis'].value_counts().reindex(GROUPS)
    rows = []

    def cont_row(label, col):
        vals = [median_iqr(df.loc[df['diagnosis'] == g, col]) for g in GROUPS]
        samples = [df.loc[df['diagnosis'] == g, col].dropna() for g in GROUPS]
        rows.append((label, vals, stats.kruskal(*samples).pvalue))

    def header_row(label):
        rows.append((label, [''] * len(GROUPS), None))

    cont_row('BMI (kg/m²)', 'bmi')
    for muscle, cfg in MUSCLES.items():
        header_row(muscle)
        cont_row('Skinfold caliper (cm)', cfg['caliper'])
        header_row('Deepest usable optode, n (%)')
        deepest = get_deepest_optodes(df, cfg['optode_cols'])
        counts = (pd.crosstab(deepest, df['diagnosis']).reindex(index=list(DEPTH_LABELS), columns=GROUPS, fill_value=0))
        pct = (counts / n * 100).round(0).astype(int)
        for d, lab in DEPTH_LABELS.items():
            rows.append((lab, [f'{counts.loc[d, g]} ({pct.loc[d, g]}%)' for g in GROUPS], None))
        if counts.loc[0].sum() > 0:
            print(f"{muscle}: participants with no usable optode:")
            print(df.loc[deepest == 0, 'basename'].tolist())

    out = pd.DataFrame([r[1] for r in rows], columns=[f'{g} (n={n[g]})' for g in GROUPS], index=[r[0] for r in rows])
    tested = [i for i, r in enumerate(rows) if r[2] is not None]
    adj = dict(zip(tested, bh_adjust([rows[i][2] for i in tested])))
    out['p'] = [pval_str(adj[i]) if i in adj else '' for i in range(len(rows))]
    out.index.name = ''
    return out

# ------------------------------

def select_qc_optode(clean_optodes_marked, basename, muscle_prefix, suffix='_A6MCT'):
    qc_row = clean_optodes_marked[clean_optodes_marked['basename'] == f'{basename}{suffix}']
    if qc_row.empty:
        return None
    row = qc_row.iloc[0]
    for opt_num in (3, 2, 1):
        if bool(row[f'{muscle_prefix}_optode_{opt_num}']):
            return opt_num
    return None

# -------------------------------

def apply_caliper_exclusion(clean_optodes, max_caliper=3.0):
    # exclude optodes if caliper measurement exceeds 3cm (=ATT>1.5)
    out = clean_optodes.copy()
    for m, cal in {'tri': 'triceps_caliper', 'bi': 'biceps_caliper'}.items():
        out.loc[out[cal] > max_caliper, [f'{m}_optode_{i}' for i in (1, 2, 3)]] = False
    return out

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (np.sum(a[:, None] > b) - np.sum(a[:, None] < b)) / (len(a) * len(b))

# -------------------------------

def load_nirs_dfs(all_NIRS_files, participant_info):
    all_NIRS_dfs = []
    for f in all_NIRS_files:
        basename = os.path.basename(f).replace('_A6MCT.csv', '')
        nirs_df = pd.read_csv(f)
        nirs_df['basename'] = basename
        db_id = basename.split('_')[0]
        nirs_df['subject_id'] = db_id
        nirs_df['diagnosis'] = participant_info.loc[participant_info['participant_id'] == db_id, 'clinical_diagnosis'].values[0]
        all_NIRS_dfs.append(nirs_df)
    return all_NIRS_dfs

# -------------------------------

def plot_combined_trajectories_qc(all_NIRS_dfs, clean_optodes_marked, muscles=('tri', 'bi'), t_range=(-30, 450)):

    def _plot_mean_sem(ax, resampled, common_t, min_n, label_fn):
        for dx, color in DIAGNOSIS_COLORS.items():
            traces = resampled.get(dx, [])
            arr = np.vstack(traces)
            n_valid = np.sum(~np.isnan(arr), axis=0)
            mean_trace = np.nanmean(arr, axis=0)
            sem_trace = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
            keep = n_valid >= min_n
            ax.plot(common_t[keep], mean_trace[keep], color=color, lw=2.5, label=label_fn(dx, len(traces)), zorder=3)
            ax.fill_between(common_t[keep], (mean_trace - sem_trace)[keep], (mean_trace + sem_trace)[keep], color=color, alpha=0.2, edgecolor='none', zorder=2)

    def _resample_traces(all_NIRS_dfs, col_name_fn, common_t, tx_by_basename):
        #only includes basenames in tx_by_basename (QC-passed optodes) 
        resampled = {dx: [] for dx in DIAGNOSIS_COLORS.keys()}
        for df in all_NIRS_dfs:
            basename = df['basename'].iloc[0]
            if basename not in tx_by_basename:
                continue
            col_name = col_name_fn(basename)
            if col_name not in df.columns:
                continue
            dx = df['diagnosis'].iloc[0]
            sub = df[['t', col_name]].dropna()
            resampled[dx].append(np.interp(common_t, sub['t'].values, sub[col_name].values, left=np.nan, right=np.nan))
        return resampled

    def _set_phase_xticks(ax, rest_start, exercise_end, recovery_end, n_ex_ticks=6):
        minute = exercise_end / n_ex_ticks
        tick_pos = [rest_start / 2] + [minute * (i + 1) for i in range(n_ex_ticks)] + [(exercise_end + recovery_end) / 2]
        tick_labels = ['Rest'] + [str(i + 1) for i in range(n_ex_ticks)] + ['Recovery']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=10)
        ax.axvspan(rest_start, 0, color='gray', alpha=0.08, zorder=0)
        ax.axvline(0, color='gray', ls=':', lw=0.8, alpha=0.8, zorder=2.5)
        ax.axvspan(exercise_end, recovery_end, color='gray', alpha=0.08, zorder=0)
        ax.axvline(exercise_end, color='gray', ls=':', lw=0.8, alpha=0.6, zorder=2.5)

    common_t = np.linspace(*t_range, 250)
    metrics = ['O2Hb', 'HHb', 'TSI%']
    fig, axes = plt.subplots(len(metrics), len(muscles), figsize=(12, 12), squeeze=False, sharex=True)

    for col, muscle in enumerate(muscles):
        tx_by_basename = {}
        for df in all_NIRS_dfs:
            basename = df['basename'].iloc[0]
            tx = select_qc_optode(clean_optodes_marked, basename, muscle)
            if tx is not None:
                tx_by_basename[basename] = tx

        for row_i, metric in enumerate(metrics):
            ax = axes[row_i, col]

            if metric == 'TSI%': # one column only, not optode-specific
                col_name_fn = lambda b, m=muscle: f'[{m}] TSI%'
            else:
                col_name_fn = lambda b, m=muscle, met=metric: f'[{m}] Rx1-Tx{tx_by_basename[b]} {met}'
            resampled = _resample_traces(all_NIRS_dfs, col_name_fn, common_t, tx_by_basename)

            label_fn = (lambda dx, n: f'{dx} (n={n})') if row_i == 0 else (lambda dx, n: dx)
            _plot_mean_sem(ax, resampled, common_t, 3, label_fn)
            _set_phase_xticks(ax, rest_start=t_range[0], exercise_end=366, recovery_end=t_range[1])

            if row_i == 0:
                ax.set_title({'tri': 'Triceps', 'bi': 'Biceps'}.get(muscle, muscle), fontweight='bold', fontsize=14)
            if row_i == len(metrics) - 1:
                ax.set_xlabel('A6MCT (mins)', fontsize=12)

            if metric == 'TSI%':
                ax.set_ylabel('TSI%', fontsize=12)
                for y in range(50, 76, 5):
                    ax.axhline(y, color='lightgray', lw=0.65, ls='--', zorder=0)
                ax.set_ylim(50, 75)
            else:
                ax.set_ylabel(f'{metric} (ΔµM)', fontsize=12)
                if metric == 'O2Hb':
                    ax.set_ylim(-8, 16)
                    for y in range(-5, 17, 4):
                        ax.axhline(y, color='lightgray', lw=0.65, ls='--', zorder=0)
                elif metric == 'HHb':
                    y_min, y_max = (int(v) for v in ax.get_ylim())
                    for y in range(y_min, y_max-2, 2):
                        ax.axhline(y, color='lightgray', lw=0.65, ls='--', zorder=0)

    for col in range(len(muscles)):
        axes[0, col].legend(fontsize=10, loc='upper left')
    fig.tight_layout()
    fig.savefig('outputs/figures/nirs_trajectories.png', dpi=600)
    plt.close(fig)
    return fig

# -------------------------------

def build_cpet_nirs_dfs(all_CPET_files, all_NIRS_files, clean_optodes_marked):
    cpet_by_basename = {os.path.splitext(os.path.basename(f))[0]: f for f in all_CPET_files}
    nirs_by_basename = {os.path.splitext(os.path.basename(f))[0]: f for f in all_NIRS_files}
    cpet_nirs_dfs = {}

    for basename, cpet_fp in cpet_by_basename.items():
        participant_id = basename.split('_')[0]
        cpet_df = pd.read_csv(cpet_fp)
        nirs_df = pd.read_csv(nirs_by_basename[basename])

        rename_map = {}
        for m in ('tri', 'bi'):
            opt = select_qc_optode(clean_optodes_marked, basename, m, '')
            if opt is None:
                continue
            for sig in ('HHb', 'O2Hb'):
                src = f'[{m}] Rx1-Tx{opt} {sig}'
                if src in nirs_df.columns:
                    rename_map[src] = f'{sig}_{m}'

        nirs_sel = nirs_df[['t'] + list(rename_map)].rename(columns=rename_map)
        cpet_vo2_only = cpet_df[['t', 'Phase', 'VO2']] if 'VO2' in cpet_df.columns else cpet_df[['t', 'Phase']]
        cpet_nirs_dfs[participant_id] = pd.merge(cpet_vo2_only, nirs_sel, on='t', how='inner', validate='one_to_one')
    return cpet_nirs_dfs

# -------------------------------

def calculate_TSI_vars(nirs_df):

    def find_tsi_slope_window(t, y, t0, min_search_sec=2, max_search_sec=30, min_amplitude=0.0):
        """
        locate dominant initial TSI excursion around t0
        - searching from t0 - min_search_sec to t0 + max_search_sec (default: 2s before to 30s after), direction-agnostic
        - interpolates over brief NaN dropouts before smoothing (savgol_filter otherwise propagates NaN across its whole window)
        - takes the global max and global min of the smoothed trace in that window; whichever occurs first is the start of the excursion, the other is the end, favors the largest-amplitude swing in the window
        """
        t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
        window_mask = (t >= t0 - min_search_sec) & (t <= t0 + max_search_sec)
        t_win, y_raw = t[window_mask], y[window_mask]

        valid = ~np.isnan(y_raw)
        if valid.sum() < 3:
            return None, None
        if not valid.all():
            y_raw = np.interp(t_win, t_win[valid], y_raw[valid])
        n = len(y_raw)
        w = min(11, n if n % 2 == 1 else n - 1)
        y_win = savgol_filter(y_raw, w, 2)

        i_max, i_min = np.argmax(y_win), np.argmin(y_win)
        if i_max == i_min:
            return None, None
        i_ext1, i_ext2 = (i_max, i_min) if i_max < i_min else (i_min, i_max)

        dy = abs(y_win[i_ext2] - y_win[i_ext1])
        if dy < min_amplitude:
            return None, None
        return t_win[i_ext1], t_win[i_ext2]

    def excursion_slope(nirs_df, tsi_col, t0):
        t1, t2 = find_tsi_slope_window(nirs_df['t'].values, nirs_df[tsi_col].values, t0)
        if t1 is None:
            return np.nan
        window = nirs_df[(nirs_df['t'] >= t1) & (nirs_df['t'] <= t2)][['t', tsi_col]].dropna()
        return np.polyfit(window['t'], window[tsi_col], 1)[0] if len(window) >= 2 else np.nan

    exercise_df = nirs_df[nirs_df['Phase'] == 'EXERCISE']
    t0 = exercise_df['t'].iloc[0]
    recovery_df = nirs_df[nirs_df['Phase'] == 'RECOVERY']
    t0_rec = recovery_df['t'].iloc[0] if not recovery_df.empty else None

    results = {}
    for muscle_prefix in ['tri', 'bi']:
        tsi_col = f'[{muscle_prefix}] TSI%'
        if tsi_col not in nirs_df.columns:
            continue
        tsi_values = nirs_df[tsi_col].dropna()
        results[muscle_prefix] = {
            'tsi_ex_max': exercise_df[tsi_col].max(),
            'tsi_baseline': tsi_values[nirs_df['t'] < 0].mean(),
            'tsi_onsetslope': excursion_slope(nirs_df, tsi_col, t0),
            'tsi_recoveryslope': excursion_slope(nirs_df, tsi_col, t0_rec)}
    return results

# -------------------------------

def build_TSI_summary_df(all_NIRS_dfs, clean_optodes_marked):
    rows = []
    for df in all_NIRS_dfs:
        basename = df['basename'].iloc[0]
        for muscle, vals in calculate_TSI_vars(df).items():
            if select_qc_optode(clean_optodes_marked, basename, muscle) is None:
                vals = {k: np.nan for k in vals}  # all 3 optodes bad -> TSI excluded
            rows.append({'basename': basename, 'subject_id': df['subject_id'].iloc[0],
                         'diagnosis': df['diagnosis'].iloc[0], 'muscle': muscle, **vals})
    return pd.DataFrame(rows)

# -------------------------------

def build_delta_nirs_table(cpet_dfs, id_to_diag, muscles=('tri', 'bi')):

    def compute_delta_peak_or_nadir(df, col, mode):
        # zero to rest mean (t < 0), then max/min during exercise
        df = df.sort_values('t')
        if col not in df.columns or df[col].isna().all():
            return np.nan
        baseline = df.loc[df['t'] < 0, col].mean()
        ex = df.loc[df['Phase'] == 'EXERCISE', col].dropna()
        return (ex.max() if mode == 'max' else ex.min()) - baseline

    rows = []
    for participant_id, df in cpet_dfs.items():
        for m in muscles:
            for metric, (sig, mode) in {'dHHb_peak': ('HHb', 'max'), 'dO2Hb_nadir': ('O2Hb', 'min')}.items():
                rows.append({'participant_id': participant_id, 'diagnosis': id_to_diag.get(participant_id),
                             'muscle': m, 'metric': metric,
                             'value': compute_delta_peak_or_nadir(df, f'{sig}_{m}', mode)})
    return pd.DataFrame(rows)

# -------------------------------

def create_nirs_combined_table(all_NIRS_dfs, cpet_nirs_dfs, id_to_diag, clean_optodes_marked, return_long=False):

    tsi_df = build_TSI_summary_df(all_NIRS_dfs, clean_optodes_marked)
    tsi_long = tsi_df.melt(id_vars=['subject_id', 'diagnosis', 'muscle'], value_vars=['tsi_baseline', 'tsi_ex_max', 'tsi_onsetslope', 'tsi_recoveryslope'], var_name='metric', value_name='value')
    hb_long = (build_delta_nirs_table(cpet_nirs_dfs, id_to_diag).rename(columns={'participant_id': 'subject_id'})[['subject_id', 'diagnosis', 'muscle', 'metric', 'value']])
    nirs_long = pd.concat([hb_long, tsi_long], ignore_index=True)
    mismatch = set(hb_long.subject_id) ^ set(tsi_long.subject_id)
    assert not mismatch, f"mismatch between HB and TSI subject IDs: {sorted(mismatch)}"

    MUSCLE_LABELS = {'bi': 'Biceps', 'tri': 'Triceps'}
    METRIC_LABELS = {'dO2Hb_nadir': 'ΔO2Hb_nadir (μM)',
                     'dHHb_peak': 'ΔHHb_peak (μM)',
                     'tsi_baseline': 'TSI_baseline (%)',
                     'tsi_onsetslope': 'TSI_onsetslope (%/s)',
                     'tsi_recoveryslope': 'TSI_recoveryslope (%/s)'}

    dxs = [d for d in ('Control', 'SMA', 'DMD') if d in nirs_long['diagnosis'].unique()]
    n_per_dx = nirs_long.groupby('diagnosis')['subject_id'].nunique()
    contrasts = [f'{a} vs. {b}' for a, b in combinations(dxs, 2)]

    rows = []
    raw_p = {c: {} for c in contrasts}
    for muscle in MUSCLE_LABELS:
        for metric in METRIC_LABELS:
            sub = nirs_long[(nirs_long['muscle'] == muscle) & (nirs_long['metric'] == metric)]
            if sub.empty:
                continue
            vals = {d: sub.loc[sub['diagnosis'] == d, 'value'].dropna() for d in dxs}
            row = {'Metric': f'{MUSCLE_LABELS[muscle]} {METRIC_LABELS[metric]}'}
            for d in dxs:
                row[f'{d} (N={n_per_dx[d]})'] = median_iqr(vals[d])
            for a, b in combinations(dxs, 2):
                enough = len(vals[a]) >= 2 and len(vals[b]) >= 2
                raw_p[f'{a} vs. {b}'][len(rows)] = (
                    mannwhitneyu(vals[a], vals[b], alternative='two-sided')[1] if enough else np.nan)
            rows.append(row)

    table = pd.DataFrame(rows)

    # BH-FDR applied within each pairwise comparison
    for contrast, ps in raw_p.items():
        table[contrast] = pd.Series(bh_adjust(list(ps.values())), index=list(ps)).round(3)

    table = table[['Metric'] + [f'{d} (N={n_per_dx[d]})' for d in dxs] + contrasts]
    return (table, nirs_long) if return_long else table

# -------------------------------

def make_function_corr_table(peak_cpet_data, nirs_long, cpet_demographics, min_n=5):
    SCORES = {'DMD': ('pul_total', 'PUL 2.0'), 'SMA': ('rulm_best', 'RULM')}
    CPET_LABELS = {'VO2_peak_L': 'Peak VO₂ (L/min)',
                   'VO2_peak_mLkgmin': 'Peak VO₂ (mL/kg/min)',
                   'pred_vo2_pct': 'Predicted Peak VO₂ (%)',
                   'peak_RER': 'Peak RER (no unit)',
                   'peak_HR': 'Peak Heart Rate (bpm)',
                   'pred_HR_pct': 'Predicted Heart Rate (%)',
                   'O2_pulse': 'O₂ pulse (mL/bpm)'}
    METRIC_LABELS = {'dO2Hb_nadir': 'ΔO2Hb_nadir (μM)',
                     'dHHb_peak': 'ΔHHb_peak (μM)',
                     'tsi_baseline': 'TSI_baseline (%)',
                     'tsi_ex_max': 'TSI_peak (%)'}
    NIRS_LABELS = {f'{metric}_{m}': f'{muscle} {label}' for m, muscle in {'bi': 'Biceps', 'tri': 'Triceps'}.items()
                   for metric, label in METRIC_LABELS.items()}

    demo = cpet_demographics[cpet_demographics['clinical_diagnosis'].isin(SCORES)]
    assert demo['participant_id'].is_unique, 'expected one CPET visit per participant'
    nirs_wide = nirs_long.assign(col=nirs_long['metric'] + '_' + nirs_long['muscle']).pivot(index='subject_id', columns='col', values='value')
    data = (demo.set_index('participant_id')[['clinical_diagnosis', 'pul_total', 'rulm_best']].join(peak_cpet_data.set_index('participant_id')[list(CPET_LABELS)]).join(nirs_wide[list(NIRS_LABELS)]))

    tables = {}
    for family, labels in {'Peak CPET': CPET_LABELS, 'Peak NIRS': NIRS_LABELS}.items():
        recs = []
        for dx, (score, score_label) in SCORES.items():
            g = data[data['clinical_diagnosis'] == dx]
            for col, label in labels.items():
                sub = g[[score, col]].dropna()
                ok = len(sub) >= min_n and sub.nunique().min() > 1
                rho, p = stats.spearmanr(sub[score], sub[col]) if ok else (np.nan, np.nan)
                recs.append({'Metric': label, 'group': f'{dx} ({score_label})', 'n': len(sub), 'ρ': rho, 'p': p})
        res = pd.DataFrame(recs)
        res['FDR p'] = res.groupby('group')['p'].transform(lambda s: bh_adjust(s.values))
        res['ρ'] = res['ρ'].map(lambda r: 'N/A' if np.isnan(r) else f'{r:.2f}')
        res['p'], res['FDR p'] = res['p'].map(pval_str), res['FDR p'].map(pval_str)
        wide = res.pivot(index='Metric', columns='group', values=['n', 'ρ', 'p', 'FDR p']).swaplevel(axis=1)
        col_order = [(f'{dx} ({score_label})', s) for dx, (_, score_label) in SCORES.items() for s in ['n', 'ρ', 'p', 'FDR p']]
        tables[family] = wide.reindex(list(labels.values()))[col_order]

    return pd.concat(tables, names=['Family', 'Metric'])

# ------------------------------------------------------------
######## HHb/VO2 RATIO
# ------------------------------------------------------------

def compute_hhb_vo2_ratio(df, hhb_col, vo2_shift_s=20, ratio_window=(20, 120), baseline_s=60, steadystate_s=60):
    # baseline = last baseline_s of rest (t < 0), steady state = last steadystate_s of exercise
    # VO2 shifted left vo2_shift_s for the phase I -> phase II delay
    df = apply_30s_rolling_average(df.sort_values('t')).set_index('t')
    ex_end = df.index[df['Phase'] == 'EXERCISE'][-1]
    bl = df.loc[-baseline_s:0]
    ss = df.loc[ex_end - steadystate_s:ex_end]

    vo2_baseline, vo2_ss = bl['VO2'].mean(), ss['VO2'].mean()
    hhb_baseline, hhb_ss = bl[hhb_col].mean(), ss[hhb_col].mean()

    df['VO2_norm'] = (df['VO2'] - vo2_baseline) / (vo2_ss - vo2_baseline) * 100
    df['HHb_norm'] = (df[hhb_col] - hhb_baseline) / (hhb_ss - hhb_baseline) * 100
    df['VO2_norm_shifted'] = df['VO2_norm'].shift(-vo2_shift_s)
    df['HHb_VO2_ratio'] = df['HHb_norm'] / df['VO2_norm_shifted'].where(df['VO2_norm_shifted'].abs() > 5) # avoid div by small #s
    ratio_mean = df.loc[ratio_window[0]:ratio_window[1], 'HHb_VO2_ratio'].mean()

    diagnostics = {'vo2_ss': vo2_ss, 'vo2_range': vo2_ss - vo2_baseline, 'hhb_range': hhb_ss - hhb_baseline}
    return df, ratio_mean, diagnostics

# -------------------------------

def build_hhb_vo2_table(cpet_nirs_dfs, id_to_diag, **ratio_kwargs):
    records = []
    for participant_id, df in cpet_nirs_dfs.items():
        for muscle in ['tri', 'bi']:
            hhb_col = f'HHb_{muscle}'
            if hhb_col not in df.columns:
                continue
            _, ratio, _ = compute_hhb_vo2_ratio(df, hhb_col=hhb_col, **ratio_kwargs)
            records.append({'participant_id': participant_id, 'diagnosis': id_to_diag.get(participant_id), 'muscle': muscle, 'HHb_VO2_ratio': ratio})
    return pd.DataFrame(records)

def build_hhb_vo2_summary_table(hhb_vo2_table):
    # BH-FDR across muscles within each pairwise comparison
    groups = ['Control', 'SMA', 'DMD']
    pairs = list(combinations(groups, 2))
    rows, ps = [], []
    for muscle in hhb_vo2_table['muscle'].unique():
        sub = hhb_vo2_table[hhb_vo2_table['muscle'] == muscle]
        vals = {g: sub.loc[sub['diagnosis'] == g, 'HHb_VO2_ratio'].dropna() for g in groups}
        rows.append({'variable': f'HHb/VO2 ({muscle})', **{g: f'{median_iqr(vals[g])} (n={len(vals[g])})' for g in groups}})
        for g1, g2 in pairs:
            ps.append(mannwhitneyu(vals[g1], vals[g2], alternative='two-sided')[1] if len(vals[g1]) and len(vals[g2]) else np.nan)
    ps = np.array(ps).reshape(len(rows), len(pairs))
    table = pd.DataFrame(rows)
    for j, (g1, g2) in enumerate(pairs):
        table[f'{g1} vs. {g2}'] = bh_adjust(ps[:, j]).round(3)
    return table

# -------------------------------

def find_ss_crossing_time(trace, target=1.0, tolerance=0.15, sustain_s=15, search_from=0, search_to=None):
    ratio = trace['HHb_VO2_ratio']
    if search_to is None:
        search_to = trace.index.max()
    ratio = ratio.loc[search_from:search_to]

    in_band = (ratio - target).abs() <= tolerance
    in_band = in_band.reindex(np.arange(search_from, search_to + 1), fill_value=False)
    vals = in_band.to_numpy()
    idx = in_band.index.to_numpy()

    for i in range(len(vals) - sustain_s + 1):
        if vals[i:i + sustain_s].all():
            return idx[i]
    return np.nan

def build_ss_crossing_table(cpet_nirs_dfs, id_to_diag, muscle='tri', **kwargs):
    hhb_col = f'HHb_{muscle}'
    records = []
    for participant_id, df in cpet_nirs_dfs.items():
        diagnosis = id_to_diag.get(participant_id)
        if diagnosis is None or hhb_col not in df.columns:
            continue
        trace, _, _ = compute_hhb_vo2_ratio(df, hhb_col=hhb_col)
        records.append({'participant_id': participant_id, 'diagnosis': diagnosis, 't_ss_crossing': find_ss_crossing_time(trace, **kwargs)})
    return pd.DataFrame(records)

def summarize_ss_crossing(crossing_df):
    rows = []
    for g in ['Control', 'SMA', 'DMD']:
        sub = crossing_df[crossing_df['diagnosis'] == g]
        reached = sub['t_ss_crossing'].dropna()
        rows.append({
            'diagnosis': g,
            'n_total': len(sub),
            'n_reached': len(reached),
            'pct_reached': round(100 * len(reached) / len(sub), 0) if len(sub) > 0 else np.nan,
            'mean_t': round(reached.mean(), 1) if len(reached) > 0 else np.nan,
            'median_t': round(reached.median(), 1) if len(reached) > 0 else np.nan,
        })
    return pd.DataFrame(rows)

# -------------------------------

def get_range_diagnostics(cpet_nirs_dfs, id_to_diag, diagnosis, muscle='tri'):
    hhb_col = f'HHb_{muscle}'
    records = []
    for participant_id, df in cpet_nirs_dfs.items():
        if id_to_diag.get(participant_id) != diagnosis or hhb_col not in df.columns:
            continue
        _, ratio, diag = compute_hhb_vo2_ratio(df, hhb_col=hhb_col)
        records.append({'participant_id': participant_id, 'ratio': ratio, 'hhb_range': diag['hhb_range'], 'vo2_range': diag['vo2_range']})
    return pd.DataFrame(records).sort_values('ratio')

def plot_range_quadrants(diagnostics_by_group, muscle='tri'):
    fig, ax = plt.subplots(figsize=(4, 4))
    for diagnosis, df in diagnostics_by_group.items():
        ax.scatter(df['vo2_range'], df['hhb_range'], color=DIAGNOSIS_COLORS[diagnosis], label=diagnosis, alpha=0.7, s=40)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('VO2 range (steady-state − baseline)', fontdict={'fontsize': 11})
    ax.set_ylabel('HHb range (steady-state − baseline)', fontdict={'fontsize': 11})
    ax.set_title({'tri': 'Triceps', 'bi': 'Biceps'}.get(muscle, muscle))
    if muscle == 'bi':
        ax.legend()
    fig.tight_layout()
    return fig

# -------------------------------

def compute_a6mct_relative_intensity(cpet_nirs_dfs, peak_cpet_data, id_to_diag):
    records = []
    for participant_id, df in cpet_nirs_dfs.items():
        # VO2 doesn't depend on muscle, so any hhb_col works here, just need vo2_ss
        hhb_col = 'HHb_tri' if 'HHb_tri' in df.columns else 'HHb_bi'
        if hhb_col not in df.columns:
            continue
        _, _, diag = compute_hhb_vo2_ratio(df, hhb_col=hhb_col)
        peak_row = peak_cpet_data[peak_cpet_data['participant_id'] == participant_id]
        if peak_row.empty:
            continue
        peak_vo2_mL = peak_row.iloc[0]['VO2_peak_L'] * 1000
        records.append({
            'participant_id': participant_id,
            'diagnosis': id_to_diag.get(participant_id),
            'a6mct_vo2_ss_L': diag['vo2_ss'] / 1000,
            'peak_vo2_L': peak_vo2_mL / 1000,
            'a6mct_pct_peak': 100 * diag['vo2_ss'] / peak_vo2_mL,
        })
    return pd.DataFrame(records)