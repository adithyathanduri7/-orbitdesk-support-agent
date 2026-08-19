
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

def draw_graph():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor('#0f172a')
    fig.patch.set_facecolor('#0f172a')

    def node(x, y, text, color='#1e3a5f',
             text_color='white', width=2.2,
             height=0.7):
        rect = mpatches.FancyBboxPatch(
            (x - width/2, y - height/2),
            width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='#38bdf8',
            linewidth=2,
            zorder=3
        )
        ax.add_patch(rect)
        ax.text(
            x, y, text,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color=text_color, zorder=4
        )

    def arrow(x1, y1, x2, y2,
              label='', color='#38bdf8'):
        ax.annotate(
            '',
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->',
                color=color,
                lw=2
            ),
            zorder=2
        )
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2
            ax.text(
                mx + 0.15, my, label,
                fontsize=8, color='#94a3b8',
                zorder=5
            )

    # Title
    ax.text(
        7, 9.5,
        'OrbitDesk Support Agent — LangGraph',
        ha='center', va='center',
        fontsize=14, fontweight='bold',
        color='#38bdf8'
    )

    ax.text(
        7, 9.1,
        'Tantrabodh AI | Thanduri Adithya',
        ha='center', va='center',
        fontsize=9, color='#94a3b8'
    )

    # START
    node(7, 8.3, 'START', '#0ea5e9')

    # TRIAGE
    node(7, 7.0,
         'TRIAGE NODE\nClassify question',
         '#1e3a5f')

    # Decision diamond
    diamond = plt.Polygon(
        [[7, 5.8], [8.2, 5.2],
         [7, 4.6], [5.8, 5.2]],
        facecolor='#292524',
        edgecolor='#38bdf8',
        linewidth=2, zorder=3
    )
    ax.add_patch(diamond)
    ax.text(
        7, 5.2,
        'Classification?',
        ha='center', va='center',
        fontsize=8, color='white', zorder=4
    )

    # RETRIEVAL
    node(7, 3.8,
         'RETRIEVAL NODE\nFAISS + Embeddings',
         '#1e3a5f')

    # GENERATION
    node(7, 2.6,
         'GENERATION NODE\nLocal LLM (TinyLlama)',
         '#1e3a5f')

    # VERIFICATION
    node(7, 1.4,
         'VERIFICATION NODE\nQuality checks',
         '#1e3a5f')

    # OUTPUT
    node(7, 0.3, 'OUTPUT / END',
         '#0ea5e9')

    # Out of scope box
    node(11, 2.6,
         'OUT OF SCOPE\nSafe response',
         '#7f1d1d', '#fca5a5')

    # Retry box
    node(3, 2.1,
         'RETRY\n(max 1)',
         '#713f12', '#fbbf24')

    # Arrows
    arrow(7, 8.0, 7, 7.35)              # start→triage
    arrow(7, 6.65, 7, 5.8)             # triage→decision
    arrow(7, 4.6, 7, 4.15)             # decision→retrieval
    arrow(7, 3.45, 7, 2.95)            # retrieval→gen
    arrow(7, 2.25, 7, 1.75)            # gen→verify
    arrow(7, 1.05, 7, 0.65)            # verify→output

    # Out of scope path
    ax.annotate('', xy=(9.9, 2.6),
                xytext=(8.2, 5.2),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#f87171', lw=2))
    ax.text(9.2, 4.2, 'out_of_scope\nescalation',
            fontsize=8, color='#f87171')

    # Retry path
    ax.annotate('', xy=(3, 2.6),
                xytext=(5.8, 5.2),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#fbbf24', lw=2))
    ax.text(3.5, 4.2, 'clarification',
            fontsize=8, color='#fbbf24')

    ax.annotate('', xy=(5.8, 2.6),
                xytext=(3, 2.35),
                arrowprops=dict(
                    arrowstyle='->',
                    color='#fbbf24', lw=2,
                    connectionstyle='arc3,rad=0.2'))
    ax.text(2.0, 2.1, 'retry',
            fontsize=8, color='#fbbf24')

    # Legend
    legend_items = [
        mpatches.Patch(
            facecolor='#1e3a5f',
            edgecolor='#38bdf8',
            label='Processing Node'),
        mpatches.Patch(
            facecolor='#0ea5e9',
            label='Entry/Exit'),
        mpatches.Patch(
            facecolor='#7f1d1d',
            edgecolor='#ef4444',
            label='Safe Failure'),
        mpatches.Patch(
            facecolor='#713f12',
            edgecolor='#f59e0b',
            label='Retry Path'),
    ]
    ax.legend(
        handles=legend_items,
        loc='lower left',
        facecolor='#1e293b',
        edgecolor='#334155',
        labelcolor='white',
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(
        'graph_diagram.png',
        dpi=150,
        bbox_inches='tight',
        facecolor='#0f172a'
    )
    print("✅ Graph diagram saved: graph_diagram.png")

if __name__ == "__main__":
    draw_graph()