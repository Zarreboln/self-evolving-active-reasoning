"""Render descriptive partial GN results from anonymous data (requires matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from verify_results import derive, ARMS


def main():
    base = Path(__file__).absolute().parent
    result = derive(json.loads((base / 'anonymous-numerics.json').read_text(encoding='utf-8')))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'gn-partial-v1'})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3), gridspec_kw={'height_ratios': [1.12, 1]})
    ax = axes[0, 0]
    for arm, color, marker in [('DC_Cumulative', '#2563a6', 'o'), ('ACE_Supervised', '#ca6620', 's')]:
        rows = sorted([x for x in result['tables'] if x['split'] == 'dev' and x['arm'] == arm],
                      key=lambda x: x['checkpoint'])
        ax.plot([x['checkpoint'] for x in rows],
                [100 * x['actor_final_successes'] / x['actor_final_assessed'] for x in rows],
                marker=marker, fillstyle='none' if arm.startswith('ACE') else 'full',
                markersize=9 if arm.startswith('ACE') else 5, color=color, label=arm.replace('_', ' '))
    ax.set(title='A  Complete dev checkpoints: final actor accuracy', xlabel='Training updates',
           ylabel='Final actor EM', xticks=[0, 3, 6], ylim=(-5, 100))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.text(3, 25, 'Both methods: 0/5 at all three checkpoints\nSame five cases reused; one trajectory per method', ha='center', fontsize=10)
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', alpha=.18)
    ax = axes[0, 1]
    ax.axis('off')
    ax.set_title('B  Final evaluation: common 13-case prefix only', loc='left')
    table_rows = []
    for arm in ARMS:
        r = next(x for x in result['tables'] if x['split'] == 'final_common_prefix' and x['arm'] == arm)
        actor = f"{r['actor_final_successes']}/{r['actor_final_assessed']}" if r['actor_final_assessed'] else f"NA ({r['n']} cases)"
        if r['actor_final_unassessed'] and r['actor_final_assessed']:
            actor += f"; {r['actor_final_unassessed']} NA"
        table_rows.append([arm.replace('_parser_repair01', ' (repair)').replace('_', ' '),
                           f"{r['native_successes']}/{r['n']}", actor])
    table = ax.table(cellText=table_rows, colLabels=['Method', 'Native success', 'Actor final EM'],
                     colWidths=[.47, .24, .29], bbox=[0, .03, 1, .85], cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#e0e4e8')
        if row == 0:
            cell.set_facecolor('#edf2f7'); cell.set_text_props(weight='bold')
    ax = axes[1, 0]
    dc = result['dc']
    ax.barh([1, 0], [100 * dc['length_limited'] / 6, 100 * dc['body_changes_after_first'] / 5],
            color=['#ca6620', '#2563a6'], height=.5)
    ax.set(yticks=[1, 0], yticklabels=['Curator outputs\nat token cap', 'Memory body changes\nafter first update'],
           xlim=(0, 110), ylim=(-.6, 1.7), title='C  DC: truncation and retained text')
    ax.text(102, 1, f"{dc['length_limited']}/6", va='center', ha='left')
    ax.text(2, 0, f"{dc['body_changes_after_first']}/5", va='center')
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.text(0, -.42, 'Version line removed for text comparison; code execution disabled.', fontsize=8)
    ax = axes[1, 1]
    ace = result['ace']
    vals = [ace[k] for k in ('initial_correct', 'post_reflection_correct', 'post_curation_correct')]
    ax.bar(range(3), [100 * x / 6 for x in vals], color=['#8795a4', '#2563a6', '#ca6620'], width=.55)
    for i, v in enumerate(vals): ax.text(i, 100 * v / 6 + 4, f'{v}/6', ha='center')
    ax.set(xticks=range(3), xticklabels=['Initial', 'After reflection', 'After memory update'],
           ylim=(0, 120), yticks=[0, 20, 40, 60, 80, 100],
           title='D  ACE: same-training-case correctness', ylabel='Final actor EM')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.text(.5, .98, 'Reflector sees the training target; these are not heldout gains.',
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    fig.suptitle('GN pilot: complete development trajectory, incomplete final comparison',
                 fontsize=16, x=.03, ha='left', y=.985)
    fig.text(.03, .937, 'PARTIAL_VERIFIED integrity/scoring only • Original run failed • Frozen Qwen2.5-7B • Actor cap 512 tokens', fontsize=10)
    fig.text(.03, .025, 'Final coverage: 97/140 recorded; 43 missing = 1 interrupted + 42 not started. Six recorded rows outside the common prefix are omitted here.\nRuntime/order-dependent missingness prevents a clean seven-method ranking. Descriptive only: no intervals, significance tests or general method-effect claim.', fontsize=9)
    fig.subplots_adjust(left=.15, right=.98, top=.86, bottom=.14, hspace=.55, wspace=.28)
    fig.savefig(base / 'partial-results.png', dpi=180)
    fig.savefig(base / 'partial-results.svg', metadata={'Date': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
