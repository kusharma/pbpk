"""
Matplotlib defaults (single-font, compact) with standardized 2-panel geometry.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- Single font (Arial) ----
PUB_FONT_SIZES = {
    "title": 7.5,     # small panel title (if you use one)
    "axes": 6.8,      # axis-label size (reduced)
    "ticks": 5.8,     # tick-label size (reduced)
    "legend": 6.2,    # legend text
}

# Transparency for mean line overlays
MEAN_LINE_ALPHA = 0.5

# Transparency for threshold line at -50 µm
THRESHOLD_LINE_ALPHA = 0.25

# ---- Standardized subplot geometry (inches) ----
# Increased axes size and reduced gap to maximize plotting area
AX_W_IN  = 2.80   # Increased from 2.60
AX_H_IN  = 2.00   # Increased from 1.80
GAP_W_IN = 0.25   # Reduced from 0.40 to bring subplots closer
M_L_IN   = 0.50   # Reduced from 0.55
M_R_IN   = 0.30   # Reduced from 0.35
M_T_IN   = 0.35   # Reduced from 0.40
M_B_IN   = 0.45   # Reduced from 0.50

FIG_W_IN = M_L_IN + AX_W_IN*2 + GAP_W_IN + M_R_IN
FIG_H_IN = M_B_IN + AX_H_IN + M_T_IN
PUB_FIGSIZE_2PANEL = (FIG_W_IN, FIG_H_IN)

# 2×2 (4-panel) figure dimensions
FIG_W_IN_4PANEL = M_L_IN + AX_W_IN*2 + GAP_W_IN + M_R_IN
FIG_H_IN_4PANEL = M_B_IN + AX_H_IN*2 + GAP_W_IN + M_T_IN
PUB_FIGSIZE_4PANEL = (FIG_W_IN_4PANEL, FIG_H_IN_4PANEL)

# Convert physical gap to subplots_adjust wspace (fraction of axes width)
WSPACE_FRAC = GAP_W_IN / AX_W_IN  # ~0.18
HSPACE_FRAC = GAP_W_IN / AX_H_IN  # ~0.22 (vertical gap)

SUBPLOT_TITLE_X_A = -0.1   # X-position for panel A (axes fraction; 0=left, <0=outside axis)
SUBPLOT_TITLE_X_B = -0.1      # X-position for panel B
SUBPLOT_TITLE_X_C = -0.1   # X-position for panel C
SUBPLOT_TITLE_X_D = -0.1   # X-position for panel D

def set_panel_title(ax, title, panel='A', y=1.06, fontsize=None, 
                    x_pos_A=None, x_pos_B=None, x_pos_C=None, x_pos_D=None, **kwargs):
    """
    Place a left-aligned panel title at a custom horizontal offset, using axes coordinates.
    panel: 'A', 'B', 'C', or 'D' (to pick the right X)
    y: vertical placement in axes fraction (default just above axes)
    fontsize: use PUB_FONT_SIZES['title'] if not specified
    x_pos_A, x_pos_B, x_pos_C, x_pos_D: override default X positions for each panel
    kwargs: passed to ax.text()
    """
    if fontsize is None:
        fontsize = PUB_FONT_SIZES['title']
    
    # Use provided x positions or fall back to defaults
    if panel == 'A':
        x = x_pos_A if x_pos_A is not None else SUBPLOT_TITLE_X_A
    elif panel == 'B':
        x = x_pos_B if x_pos_B is not None else SUBPLOT_TITLE_X_B
    elif panel == 'C':
        x = x_pos_C if x_pos_C is not None else SUBPLOT_TITLE_X_C
    elif panel == 'D':
        x = x_pos_D if x_pos_D is not None else SUBPLOT_TITLE_X_D
    else:
        x = 0.0
    
    # Make panel labels A and B bold and increase font size by 1
    weight = 'bold' if panel in ['A', 'B'] else 'normal'
    if panel in ['A', 'B']:
        fontsize = fontsize + 1  # Increase font size by 1 for A and B
    ax.text(x, y, title, fontsize=fontsize, ha='left', va='bottom', 
            transform=ax.transAxes, weight=weight, **kwargs)

# Backward compatibility: make set_left_title an alias to set_panel_title (Panel A default)
def set_left_title(ax, title, pad=None, fontsize=None, **kwargs):
    set_panel_title(ax, title, panel='A', fontsize=fontsize, **kwargs)

def set_pub_defaults():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.titlesize": PUB_FONT_SIZES["title"],
        "axes.labelsize": PUB_FONT_SIZES["axes"],
        "xtick.labelsize": PUB_FONT_SIZES["ticks"],
        "ytick.labelsize": PUB_FONT_SIZES["ticks"],
        "legend.fontsize": PUB_FONT_SIZES["legend"],
        "legend.title_fontsize": PUB_FONT_SIZES["legend"] ,

        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "lines.linewidth": 1.0,

        "figure.dpi": 600,
        "savefig.dpi": 600,
    })

def apply_standard_2panel_layout(fig, wspace=None, left=None, right=None, bottom=None, top=None):
    """
    Apply consistent margins and gap for 1×2 (horizontal) layout.
    
    Parameters:
    -----------
    wspace : float, optional
        Space between subplots as fraction of axis width. Default: WSPACE_FRAC
    left, right, bottom, top : float, optional
        Override default margins (as fraction of figure width/height)
    """
    if left is None:
        left = M_L_IN / FIG_W_IN
    if right is None:
        right = 1.0 - (M_R_IN / FIG_W_IN)
    if bottom is None:
        bottom = M_B_IN / FIG_H_IN
    if top is None:
        top = 1.0 - (M_T_IN / FIG_H_IN)
    if wspace is None:
        wspace = WSPACE_FRAC
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace)

def apply_standard_4panel_layout(fig):
    """Apply consistent margins and gaps for 2×2 layout."""
    left   = M_L_IN / FIG_W_IN_4PANEL
    right  = 1.0 - (M_R_IN / FIG_W_IN_4PANEL)
    bottom = M_B_IN / FIG_H_IN_4PANEL
    top    = 1.0 - (M_T_IN / FIG_H_IN_4PANEL)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, 
                        wspace=WSPACE_FRAC, hspace=HSPACE_FRAC)

def set_pub_2panel_wide_layout(fig):
    """
    Set inter-panel and outer panel margins for two-panel figure with extra spacing for wide y-labels/legends.
    Increases wspace (inter-panel gap) to avoid axis and title collision, while preserving the standard margins otherwise.
    """
    left   = M_L_IN / FIG_W_IN
    right  = 1.0 - (M_R_IN / FIG_W_IN)
    bottom = M_B_IN / FIG_H_IN
    top    = 1.0 - (M_T_IN / FIG_H_IN)
    # Use a larger gap than standard. E.g., wspace=0.32
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=0.32)


def time_axis_locators(total_minutes: float):
    """
    Return total hours plus major/minor locators for shared time axes.
    - ≤8 h (2 cycles)        → major=1 h,  minor=0.5 h
    - (8, 24] h (1 day)      → major=6 h,  minor=3 h
    - (24, 48] h (2 days)    → major=12 h, minor=3 h
    - >48 h (e.g., 3 days)   → major=12 h, minor=4 h
    """
    total_hours = total_minutes / 60.0
    if total_hours <= 8.0:
        major_step = 1.0
        minor_step = 0.5
    elif total_hours <= 24.0:
        major_step = 6.0
        minor_step = 3.0
    elif total_hours <= 48.0:
        major_step = 12.0
        minor_step = 3.0
    else:
        major_step = 12.0
        minor_step = 4.0
    return total_hours, MultipleLocator(major_step), MultipleLocator(minor_step)

def legend_below_plots(fig, handles, labels, *, ncol=None, bbox_extra=-0.025):
    """
    Draw a single horizontal legend below a multi-subplot figure, nicely spaced and centered just under the x-axis labels.
    Use ncol=len(labels) unless otherwise given, and bbox_extra ~ -0.06 for light gap.
    """
    if ncol is None:
        ncol = len(labels)
    leg = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_extra),
        ncol=ncol,
        fontsize=6.2,
        frameon=True,
        borderpad=0.18,
        handlelength=1.0,
        fancybox=True,
        framealpha=0.92,
        labelspacing=0.5,
        columnspacing=1.5,
        borderaxespad=0.06
    )
    return leg

def legend_in_plot_kwargs(panel='A'):
    return dict(
        fontsize=6.2,
        loc='upper right',
        ncol=2 if panel == 'A' else 3,
        frameon=True,
        borderpad=0.14,
        handlelength=0.8,
        labelspacing=0.12,
        columnspacing=0.55,
        framealpha=0.92,
    )

def set_common_time_axis(ax, t_end_hours=24.0):
    """
    Set time axis with appropriate locators based on time range.
    For 24h: major=6h, minor=3h
    For 48h: major=12h, minor=3h
    """
    total_hours, major_loc, minor_loc = time_axis_locators(t_end_hours * 60.0)
    ax.set_xlim(0, total_hours)
    ax.xaxis.set_major_locator(major_loc)
    ax.xaxis.set_minor_locator(minor_loc)
    ax.grid(True, alpha=0.2, which="both")
    ax.tick_params(axis="x", direction="out")
    ax.tick_params(axis="y", direction="out")
