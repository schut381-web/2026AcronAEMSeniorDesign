import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Enables headless figure rendering
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.interpolate import interp1d

# Paths
exp_mat = '/project/mr-suo-yang/COMMON/ARL_Project/toBrian/Spark_simulation/data/Date20250703_run4.mat'
base_dir = '/project/mr-suo-yang/COMMON/ARL_Project/toBrian/Spark_simulation/allPreviousSparkRuns/old_PK_v3.0'
output_dir = '/project/mr-suo-yang/COMMON/ARL_Project/toBrian/Spark_simulation/allPreviousSparkRuns/old_PK_v3.0/plots'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# USER CONFIGURABLE: source.in values whose cases should NOT be plotted
# A case is left out of the plots if ANY of these match in its source.in.
# =============================================================================

EXCLUDE_SOURCE_START_TIMES = [-28.75]
EXCLUDE_SOURCE_END_TIMES   = [-28.0, -18.75]
VALUE_TOL = 1e-6   # tolerance for numeric comparison

# =============================================================================
# Helper Functions
# =============================================================================

_NUM = r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)'


def read_source_in(source_path):
    """
    Parses source.in and returns (start_times, end_times) as lists of floats.
    Collects every occurrence, so files with multiple sources are handled.
    Flexibly handles whitespace, colons, or equals signs.
    Returns (None, None) if the file is missing or unreadable.
    """
    if not os.path.isfile(source_path):
        return None, None

    try:
        with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        starts = [float(v) for v in re.findall(r'source_start_time\s*[:=]?\s*' + _NUM, content, re.IGNORECASE)]
        ends = [float(v) for v in re.findall(r'source_end_time\s*[:=]?\s*' + _NUM, content, re.IGNORECASE)]
        return starts, ends
    except Exception as e:
        print(f"Error reading {source_path}: {e}")
        return None, None


def _matches_any(values, targets):
    return any(abs(v - t) < VALUE_TOL for v in values for t in targets)


def is_plot_eligible(source_starts, source_ends):
    """
    True if the case should be plotted, i.e. its source.in has NONE of:
      source_start_time = -28.75
      source_end_time   = -28  or  -18.75
    Cases with no readable source.in are kept (nothing to exclude them on).
    """
    if source_starts is None and source_ends is None:
        return True
    if _matches_any(source_starts or [], EXCLUDE_SOURCE_START_TIMES):
        return False
    if _matches_any(source_ends or [], EXCLUDE_SOURCE_END_TIMES):
        return False
    return True


def ca_at_fraction(IHR, CA, TotAHR, frac):
    """
    Robust CA @ fraction without interp1 on duplicate x.
    Translates the local helper function from MATLAB.
    """
    IHR = np.maximum(IHR, 0)
    IHR = np.maximum.accumulate(IHR)  # Equivalent to cummax
    if IHR[0] > 0:
        IHR = IHR - IHR[0]

    target = frac * TotAHR
    target = np.clip(target, IHR[0], IHR[-1])

    k_indices = np.where(IHR >= target)[0]
    if len(k_indices) == 0 or k_indices[0] == 0:
        return CA[0]

    k = k_indices[0]
    x0, x1 = IHR[k - 1], IHR[k]
    y0, y1 = CA[k - 1], CA[k]

    if x1 == x0:
        return y1
    else:
        return y0 + (y1 - y0) * (target - x0) / (x1 - x0)


def ahrrcalc(CA_rng, P_SI, V_i, gamma, CAstart, CAend, ifilt, rpm):
    """
    Apparent Heat Release Rate (AHRR) thermodynamic calculation.
    """
    dP_dCA = np.gradient(P_SI, CA_rng)
    dV_dCA = np.gradient(V_i, CA_rng)

    # First law analysis (J/deg)
    AHRR = (gamma / (gamma - 1.0)) * P_SI * dV_dCA + (1.0 / (gamma - 1.0)) * V_i * dP_dCA
    AHRR = np.nan_to_num(AHRR, nan=0.0)

    # Mask outside [CAstart, CAend]
    mask = (CA_rng >= CAstart) & (CA_rng <= CAend)
    AHRR_masked = np.where(mask, AHRR, 0.0)

    # Integrated Heat Release (IHR) & Total Heat Released (TotAHR)
    IHR = np.cumsum(AHRR_masked) * np.abs(CA_rng[1] - CA_rng[0])
    TotAHR = IHR[-1] if len(IHR) > 0 else 1.0

    SOC = CAstart
    CA50 = ca_at_fraction(IHR, CA_rng, TotAHR, 0.50)
    CA90 = ca_at_fraction(IHR, CA_rng, TotAHR, 0.90)

    return AHRR, TotAHR, IHR, SOC, CA50, CA90, None


# =============================================================================
# Grid, Display & Error Range Parameters
# =============================================================================

r = 0.25                                # degCA grid resolution
CA_rng = np.arange(-50, 50 + r, r)      # Full grid array
gamma = 1.34
rpm = 1200
CAstart = -40
CAend = 40
ifilt = 1

# --- USER CONFIGURABLE: Crank Angle Window for RMSE Error Evaluation ---
CA_err_start = -20.0                    # Start of window for error calculation [degCA]
CA_err_end   =  30.0                    # End of window for error calculation [degCA]

# Create boolean mask for the chosen error evaluation range
err_mask = (CA_rng >= CA_err_start) & (CA_rng <= CA_err_end)

# Display settings
exp_cycle_color = (0.7, 0.7, 0.7)       # Gray for individual cycles
exp_mean_style = {'color': 'k', 'linestyle': '--', 'linewidth': 2}

# =============================================================================
# Load Experiment
# =============================================================================

if os.path.exists(exp_mat):
    S = loadmat(exp_mat)
    CA_exp_all = S['CAD'].flatten()
    P_exp_all = S['Pressure']  # [Nca x Ncycles], kPa
    P_exp_mean = np.nanmean(P_exp_all, axis=1)

    # Interpolate mean exp pressure to CA_rng
    interp_func = interp1d(CA_exp_all, P_exp_mean, kind='linear', fill_value='extrapolate')
    Pi_exp_kPa = interp_func(CA_rng)
    P_SI_exp = Pi_exp_kPa * 1e3  # kPa -> Pa

    hasHRR = 'HRR' in S and S['HRR'].size > 0
    if hasHRR:
        HRR_exp_all = S['HRR']
        HRR_exp_mean = np.nanmean(HRR_exp_all, axis=1)
        HRR_exp_mean_i = interp1d(CA_exp_all, HRR_exp_mean, kind='linear', fill_value='extrapolate')(CA_rng)
    else:
        HRR_exp_all = None
        HRR_exp_mean_i = None
else:
    print(f"Warning: Experimental file {exp_mat} not found. Generating placeholder synthetic exp data.")
    CA_exp_all = CA_rng
    P_exp_all = np.tile(100 + 4000 * np.exp(-((CA_rng - 5)**2) / 100), (10, 1)).T
    P_exp_mean = np.mean(P_exp_all, axis=1)
    Pi_exp_kPa = P_exp_mean
    P_SI_exp = Pi_exp_kPa * 1e3
    hasHRR = False
    HRR_exp_all = None
    HRR_exp_mean_i = None

# =============================================================================
# Recursive Search: Walk Through All Subdirectories for Run Folders
# (No cases are excluded here; every thermo.out found is loaded.)
# =============================================================================

cases = []

# Baseline case definition (always plotted)
baseline_path = '/project/mr-suo-yang/COMMON/ARL_Project/toBrian/Spark_simulation/allPreviousSparkRuns/Case02_V3.0/thermo.out'
baseline_label = 'Baseline (Case02_V3.0)'
cases.append({
    'filepath': baseline_path,
    'raw_label': baseline_label,
    'is_baseline': True,
    'plot_eligible': True
})

seen_paths = {os.path.normpath(baseline_path)}

if os.path.exists(base_dir):
    print(f"Recursively scanning directory tree starting at: {base_dir}")

    for root, dirs, files in os.walk(base_dir):
        if 'thermo.out' in files:
            thermo_path = os.path.normpath(os.path.join(root, 'thermo.out'))
            if thermo_path in seen_paths:
                continue
            seen_paths.add(thermo_path)

            rel_path_label = os.path.relpath(root, base_dir)  # e.g., "GroupA/run_01"
            src_starts, src_ends = read_source_in(os.path.join(root, 'source.in'))
            eligible = is_plot_eligible(src_starts, src_ends)

            if src_starts is None and src_ends is None:
                print(f"Note: {rel_path_label} has no readable source.in; kept for plotting.")
            elif not eligible:
                print(f"Not plotting {rel_path_label}: source.in has "
                      f"start={src_starts}, end={src_ends}")

            cases.append({
                'filepath': thermo_path,
                'raw_label': rel_path_label,
                'is_baseline': False,
                'plot_eligible': eligible,
                'source_start_times': src_starts,
                'source_end_times': src_ends
            })
else:
    print(f"Error: Base directory {base_dir} does not exist.")

n_found = len(cases) - 1
n_eligible = sum(1 for c in cases if not c['is_baseline'] and c['plot_eligible'])
print(f"\nTotal run folders discovered: {n_found}")
print(f"Run folders eligible for plotting (source.in filter): {n_eligible}")

# =============================================================================
# Load Simulation Data & Calculate Errors Over Specified Range
# =============================================================================

sims = []
for case in cases:
    filepath = case['filepath']
    label = case['raw_label']

    try:
        T = np.loadtxt(filepath)
        if T.shape[1] < 8:
            raise ValueError(f"Expected >= 8 columns; got {T.shape[1]} in {filepath}")

        # Clean non-finite rows
        valid_mask = np.isfinite(T[:, 0]) & np.isfinite(T[:, 1]) & np.isfinite(T[:, 7])
        T = T[valid_mask]

        if T.shape[0] == 0:
            raise ValueError("No valid finite data points found in file.")

        CA_raw = T[:, 0]
        P_raw = T[:, 1]  # MPa
        V_raw = T[:, 7]  # m^3

        # Handle duplicates & sort
        _, unique_idx = np.unique(CA_raw, return_index=True)
        unique_idx = np.sort(unique_idx)
        CA_s = CA_raw[unique_idx]
        P_MPa = P_raw[unique_idx]
        V_s = V_raw[unique_idx]

        sort_idx = np.argsort(CA_s)
        CA_s = CA_s[sort_idx]
        P_MPa = P_MPa[sort_idx]
        V_s = V_s[sort_idx]

        # Interpolation to grid
        Pi_MPa = interp1d(CA_s, P_MPa, kind='linear', fill_value='extrapolate')(CA_rng)
        Vi = interp1d(CA_s, V_s, kind='linear', fill_value='extrapolate')(CA_rng)
        P_SI_sim = Pi_MPa * 1e6  # MPa -> Pa
        PkPa_sim = P_SI_sim * 1e-3

        # Heat release calculation
        AHRR, TotAHR, IHR, SOC, CA50, CA90, _ = ahrrcalc(
            CA_rng, P_SI_sim, Vi, gamma, CAstart, CAend, ifilt, rpm
        )
        IHR = np.maximum(IHR, 0)
        CA10 = ca_at_fraction(IHR, CA_rng, TotAHR, 0.10)

        # Pressure RMSE (kPa) evaluated ONLY over [CA_err_start, CA_err_end]
        rmse_P = np.sqrt(np.mean((PkPa_sim[err_mask] - Pi_exp_kPa[err_mask]) ** 2))

        sims.append({
            'raw_label': label,
            'filepath': filepath,
            'is_baseline': case['is_baseline'],
            'plot_eligible': case['plot_eligible'],
            'P_Pa': P_SI_sim,
            'PkPa': PkPa_sim,
            'AHRR': AHRR,
            'IHR': IHR,
            'SOC': SOC,
            'CA50': CA50,
            'CA90': CA90,
            'Vi': Vi,
            'TotAHR': TotAHR,
            'CA10': CA10,
            'rmse_P': rmse_P
        })
    except Exception as me:
        print(f"Failed to read {filepath}\n -> {me}")

if len(sims) == 0:
    print("Error: No simulation files could be loaded. Populating mock data for preview...")
    Vi_mock = 0.0005 - 0.0003 * np.cos(np.radians(CA_rng))
    for i in range(45):
        P_mock = 100 + (3500 + i * 20) * np.exp(-((CA_rng - (2 + i * 0.1))**2) / 80)
        sims.append({
            'raw_label': f'subfolder/run_{i+1:02d}',
            'filepath': f'/mock/path_{i+1}',
            'is_baseline': False,
            'plot_eligible': True,
            'P_Pa': P_mock * 1e3,
            'PkPa': P_mock,
            'AHRR': np.gradient(P_mock, CA_rng),
            'IHR': np.cumsum(np.gradient(P_mock, CA_rng)),
            'SOC': -40, 'CA50': 8.0, 'CA90': 20.0,
            'Vi': Vi_mock, 'TotAHR': 500, 'CA10': 2.0,
            'rmse_P': 100.0 + i * 12.5
        })

# =============================================================================
# Calculate Experimental AHRR Reference & AHRR RMSE Over Range
# =============================================================================

Vi_exp_ref = sims[0]['Vi']
AHRR_exp_th, TotAHR_exp, IHR_exp, SOC_exp, CA50_exp, CA90_exp, _ = ahrrcalc(
    CA_rng, P_SI_exp, Vi_exp_ref, gamma, CAstart, CAend, ifilt, rpm
)

exp_ahrr_ref = HRR_exp_mean_i if (hasHRR and HRR_exp_mean_i is not None) else AHRR_exp_th

for sim in sims:
    # AHRR RMSE (J/deg) evaluated ONLY over [CA_err_start, CA_err_end]
    sim['rmse_AHRR'] = np.sqrt(np.mean((sim['AHRR'][err_mask] - exp_ahrr_ref[err_mask]) ** 2))

# Separate baseline from test simulation runs
baseline_sim = next((s for s in sims if s['is_baseline']), None)
test_sims_all = [s for s in sims if not s['is_baseline']]

# Only cases passing the source.in filter are ranked and plotted
test_sims_plot = [s for s in test_sims_all if s['plot_eligible']]
print(f"\nLoaded {len(test_sims_all)} test cases; {len(test_sims_plot)} pass the source.in filter and will be ranked/plotted.")

# =============================================================================
# Plotting Function with Error Window Highlight
# =============================================================================

def plot_top_cases(test_cases, metric_key, metric_unit, filename_prefix, title_prefix):
    max_plots_per_fig = 10
    test_chunks = [test_cases[i:i + max_plots_per_fig] for i in range(0, len(test_cases), max_plots_per_fig)] if test_cases else [[]]

    linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
    linewidths = [2.5, 2.0, 1.8, 2.2, 1.6]

    for chunk_idx, test_chunk in enumerate(test_chunks, start=1):
        chunk_cmap = plt.get_cmap('tab10')
        chunk_sims_info = []

        # Add baseline first if present
        if baseline_sim:
            chunk_sims_info.append({
                'sim': baseline_sim,
                'color': (0.0, 0.0, 0.0),  # Solid Black
                'style': '-',
                'linewidth': 3.0,
                'alpha': 1.0,
                'label': baseline_sim['raw_label']
            })

        for sim_i, sim in enumerate(test_chunk):
            global_rank = (chunk_idx - 1) * max_plots_per_fig + sim_i + 1
            color = chunk_cmap(sim_i % 10)
            style = linestyles[sim_i % len(linestyles)]
            lw = linewidths[sim_i % len(linewidths)]
            label = f"#{global_rank}: {sim['raw_label']} (RMSE: {sim[metric_key]:.2f} {metric_unit})"

            chunk_sims_info.append({
                'sim': sim,
                'color': color,
                'style': style,
                'linewidth': lw,
                'alpha': 0.85,
                'label': label
            })

        # --- Plot 1: Pressure Overlay ---
        fig1, ax1 = plt.subplots(figsize=(12, 9))
        if P_exp_all.ndim > 1 and P_exp_all.shape[1] > 1:
            ax1.plot(CA_exp_all, P_exp_all, color=exp_cycle_color, linewidth=0.75, alpha=0.5)
        ax1.plot(CA_rng, Pi_exp_kPa, label='Expt. mean P', **exp_mean_style)

        ax1.axvspan(CA_err_start, CA_err_end, color='yellow', alpha=0.12,
                    label=f'Error Window [{CA_err_start:.0f}, {CA_err_end:.0f}] degCA')

        for info in chunk_sims_info:
            sim = info['sim']
            ax1.plot(CA_rng, sim['PkPa'], linestyle=info['style'], linewidth=info['linewidth'],
                     color=info['color'], alpha=info['alpha'], label=info['label'])

        ax1.set_xlim([-50, 50])
        ax1.set_xticks(np.arange(-50, 51, 10))
        ax1.set_ylim([0, 5000])
        ax1.set_yticks(np.arange(0, 5001, 1000))
        ax1.set_xlabel('Crank angle [degCA]', fontsize=18, fontweight='bold')
        ax1.set_ylabel('Pressure [kPa]', fontsize=18, fontweight='bold')
        ax1.tick_params(labelsize=16)
        ax1.grid(True)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_title(f'Pressure: {title_prefix} - Part {chunk_idx}', fontsize=16)
        fig1.tight_layout()
        fig1.savefig(os.path.join(output_dir, f'{filename_prefix}_pressure_part{chunk_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # --- Plot 2: AHRR Overlay ---
        fig2, ax2 = plt.subplots(figsize=(12, 9))
        if hasHRR and HRR_exp_all is not None:
            ax2.plot(CA_exp_all, HRR_exp_all, color=exp_cycle_color, linewidth=0.75, alpha=0.5)
            ax2.plot(CA_rng, HRR_exp_mean_i, label='Expt. HRR mean', **exp_mean_style)

        ax2.axvspan(CA_err_start, CA_err_end, color='yellow', alpha=0.12,
                    label=f'Error Window [{CA_err_start:.0f}, {CA_err_end:.0f}] degCA')

        for info in chunk_sims_info:
            sim = info['sim']
            ax2.plot(CA_rng, sim['AHRR'], linestyle=info['style'], linewidth=info['linewidth'],
                     color=info['color'], alpha=info['alpha'], label=info['label'])

        ax2.set_xlim([-50, 50])
        ax2.set_xticks(np.arange(-50, 51, 10))
        ax2.set_ylim([-10, 20])
        ax2.set_yticks(np.arange(-10, 21, 5))
        ax2.set_xlabel('Crank angle [degCA]', fontsize=18, fontweight='bold')
        ax2.set_ylabel('AHRR [J/deg]', fontsize=18, fontweight='bold')
        ax2.tick_params(labelsize=16)
        ax2.grid(True)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.set_title(f'AHRR: {title_prefix} - Part {chunk_idx}', fontsize=16)
        fig2.tight_layout()
        fig2.savefig(os.path.join(output_dir, f'{filename_prefix}_ahrr_part{chunk_idx}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)


# =============================================================================
# 1. Rank & Plot Top 30 Cases by Lowest Pressure Error
# =============================================================================

test_sims_P_best = sorted(test_sims_plot, key=lambda s: s['rmse_P'])[:30]

print(f"\n--- Top 30 Cases with Lowest Pressure Error in [{CA_err_start}, {CA_err_end}] degCA Window ---")
for rank, sim in enumerate(test_sims_P_best, 1):
    print(f"#{rank}: {sim['raw_label']} (Pressure RMSE: {sim['rmse_P']:.2f} kPa)")

plot_top_cases(
    test_cases=test_sims_P_best,
    metric_key='rmse_P',
    metric_unit='kPa',
    filename_prefix='top30_pressure_error',
    title_prefix=f'Top 30 Lowest Pressure Error Cases [{CA_err_start:.0f} to {CA_err_end:.0f} degCA]'
)

# =============================================================================
# 2. Rank & Plot Top 30 Cases by Lowest AHRR Error
# =============================================================================

test_sims_AHRR_best = sorted(test_sims_plot, key=lambda s: s['rmse_AHRR'])[:30]

print(f"\n--- Top 30 Cases with Lowest AHRR Error in [{CA_err_start}, {CA_err_end}] degCA Window ---")
for rank, sim in enumerate(test_sims_AHRR_best, 1):
    print(f"#{rank}: {sim['raw_label']} (AHRR RMSE: {sim['rmse_AHRR']:.2f} J/deg)")

plot_top_cases(
    test_cases=test_sims_AHRR_best,
    metric_key='rmse_AHRR',
    metric_unit='J/deg',
    filename_prefix='top30_ahrr_error',
    title_prefix=f'Top 30 Lowest AHRR Error Cases [{CA_err_start:.0f} to {CA_err_end:.0f} degCA]'
)

# =============================================================================
# 3. Rank & Plot 10 Worst Cases by Pressure Error (Highest RMSE)
# =============================================================================

test_sims_P_worst = sorted(test_sims_plot, key=lambda s: s['rmse_P'], reverse=True)[:10]

print(f"\n--- 10 Worst Cases with Highest Pressure Error in [{CA_err_start}, {CA_err_end}] degCA Window ---")
for rank, sim in enumerate(test_sims_P_worst, 1):
    print(f"Worst #{rank}: {sim['raw_label']} (Pressure RMSE: {sim['rmse_P']:.2f} kPa)")

plot_top_cases(
    test_cases=test_sims_P_worst,
    metric_key='rmse_P',
    metric_unit='kPa',
    filename_prefix='worst10_pressure_error',
    title_prefix=f'10 Worst (Highest) Pressure Error Cases [{CA_err_start:.0f} to {CA_err_end:.0f} degCA]'
)

# =============================================================================
# 4. Rank & Plot 10 Worst Cases by AHRR Error (Highest RMSE)
# =============================================================================

test_sims_AHRR_worst = sorted(test_sims_plot, key=lambda s: s['rmse_AHRR'], reverse=True)[:10]

print(f"\n--- 10 Worst Cases with Highest AHRR Error in [{CA_err_start}, {CA_err_end}] degCA Window ---")
for rank, sim in enumerate(test_sims_AHRR_worst, 1):
    print(f"Worst #{rank}: {sim['raw_label']} (AHRR RMSE: {sim['rmse_AHRR']:.2f} J/deg)")

plot_top_cases(
    test_cases=test_sims_AHRR_worst,
    metric_key='rmse_AHRR',
    metric_unit='J/deg',
    filename_prefix='worst10_ahrr_error',
    title_prefix=f'10 Worst (Highest) AHRR Error Cases [{CA_err_start:.0f} to {CA_err_end:.0f} degCA]'
)

print(f"\nAll plot series (Best 30 & Worst 10) saved successfully to: {output_dir}")
