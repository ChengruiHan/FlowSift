"""Render the README overview directly from the immutable seed-101 CSV tables."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / 'experiments/seed101/report'

def read(name):
    with (EVIDENCE / name).open() as handle:
        return list(csv.DictReader(handle))

online = {r['method']: r for r in read('online_per_seed.csv') if r['method'] == 'sft' or r.get('stage') == 'formal'}
costs = {r['method']: r for r in read('system_cost.csv') if r['run_id'].startswith('formal_')}
methods = ['sft', 'full10', 'late5', 'uniform5']
labels = ['SFT', 'Full-10', 'Late-5', 'Uniform-5']
colors = ['#94A3B8', '#466C99', '#D89542', '#268579']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none', 'svg.hashsalt': 'flowsift-seed101'})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), gridspec_kw={'width_ratios': [1.05, 1]})
fig.patch.set_facecolor('#FAFBFD')
for ax in axes:
    ax.set_facecolor('#FAFBFD')
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.tick_params(axis='y', length=0, pad=9)
    ax.tick_params(axis='x', colors='#64748B')
    ax.grid(axis='x', color='#E2E8F0', linewidth=.7)
    ax.set_axisbelow(True)

ax = axes[0]
values = [100 * float(online[m]['ode_success_rate']) for m in methods]
ax.barh(labels, values, color=colors, height=.56)
for y, v in enumerate(values):
    ax.text(v + 1.5, y, f'{v:.1f}%  ({round(v * 2)}/200)', va='center', fontsize=10, color='#1E293B')
ax.invert_yaxis()
ax.set_xlim(0, 103)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('Final ODE success (%)', labelpad=10)
ax.set_title('A   Task performance', loc='left', pad=18, fontweight='bold', color='#0F172A')

ax = axes[1]
train_methods = methods[1:]
bottom = [0.0] * 3
for label, keys, color in [
    ('Rollout', ['rollout_seconds'], '#CBD5E1'),
    ('Learner', ['learner_seconds'], '#466C99'),
    ('Other loop work', ['diagnostic_seconds', 'checkpoint_seconds', 'other_seconds'], '#D89542'),
]:
    values = [sum(float(costs[m][key]) for key in keys) / 60 for m in train_methods]
    ax.barh(labels[1:], values, left=bottom, color=color, height=.56, label=label)
    bottom = [a + b for a, b in zip(bottom, values)]
for y, m in enumerate(train_methods):
    elapsed = float(costs[m]['train_elapsed_seconds']) / 60
    ax.text(elapsed + 1.4, y, f'{elapsed:.1f} min', va='center', fontsize=10, color='#1E293B')
ax.invert_yaxis()
ax.set_xlim(0, 123)
ax.xaxis.set_major_locator(MultipleLocator(30))
ax.set_xlabel('Training-loop time (minutes)', labelpad=10)
ax.set_title('B   Training cost', loc='left', pad=18, fontweight='bold', color='#0F172A')
ax.legend(loc='upper center', bbox_to_anchor=(.48, -.30), ncol=3, frameon=False, fontsize=9,
          handlelength=1.2, columnspacing=1.2)
fig.subplots_adjust(left=.075, right=.97, top=.78, bottom=.26, wspace=.34)
fig.text(.075, .96, 'FlowSift | Full action generation, selective learner rescoring',
         fontsize=15, fontweight='bold', color='#0F172A', va='top')
fig.text(.075, .04, 'LIBERO-Object · training seed 101 · 300 attempted groups per method\n'
         'Single-seed observations. Loop time excludes initialization and final evaluation.',
         fontsize=9, color='#64748B', va='bottom', linespacing=1.6)
svg = HERE / 'results_overview.svg'
fig.savefig(svg, facecolor=fig.get_facecolor(), metadata={'Date': None})
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
fig.savefig(HERE / 'results_overview.png', dpi=180, facecolor=fig.get_facecolor())
plt.close(fig)
