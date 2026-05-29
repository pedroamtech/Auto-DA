"""
2_view_cluster.py — Visualizador 3D de parámetros de cámara

Mejoras sobre la versión anterior:
  · Todas las cámaras como scatter, coloreadas continuamente por pitch.
  · Frustum con FOV real: ancho y alto derivados de focal_x/y + principal_x/y.
  · Tamaño del frustum proporcional a la altura relativa de cada representante.
  · Etiqueta de cada frustum muestra pitch, altura y focal_y.
  · Panel lateral con violines de distribución por cluster:
      pitch (°) · altura relativa · focal_y (px)
  · Tabla de estadísticas: N, pitch medio, altura media, focal_y media.
"""

import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tkinter import filedialog, Tk, simpledialog


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ask_clusters():
    root = Tk(); root.withdraw(); root.attributes('-topmost', True)
    n = simpledialog.askinteger(
        "Clusters", "Número de clusters:",
        parent=root, minvalue=1, maxvalue=50, initialvalue=5
    )
    root.destroy()
    return n or 5


def _to_plot(pts):
    """World → plot coords (flip Y y Z para convenio derecho)."""
    if pts.ndim == 1:
        return np.array([pts[0], -pts[1], -pts[2]])
    r = pts.copy().astype(float)
    r[:, 1] *= -1; r[:, 2] *= -1
    return r


def _frustum_vertices(pos, R_wc, fx, fy, px, py, scale):
    """
    Vértices del frustum en espacio plot.
    El ancho y alto en la base son proporcionales al FOV real:
      half_w = (px / fx) * scale  →  tan(fov_x/2) * scale
      half_h = (py / fy) * scale  →  tan(fov_y/2) * scale
    """
    hw = (px / max(fx, 1)) * scale
    hh = (py / max(fy, 1)) * scale
    local = np.array([
        [0,   0,   0     ],   # apex (centro óptico)
        [ hw,  hh,  scale],   # esquina sup-der
        [-hw,  hh,  scale],   # esquina sup-izq
        [-hw, -hh,  scale],   # esquina inf-izq
        [ hw, -hh,  scale],   # esquina inf-der
    ]).T
    world = (R_wc @ local).T + pos
    return _to_plot(world)


def _violin(ax, data_list, labels, colors, title, ylabel):
    """Violin plot con mediana resaltada."""
    non_empty = [(d, l, c) for d, l, c in zip(data_list, labels, colors) if len(d) > 0]
    if not non_empty:
        return
    data_e, labels_e, colors_e = zip(*non_empty)
    parts = ax.violinplot(data_e, positions=range(len(data_e)),
                          showmedians=True, showextrema=True, widths=0.7)
    for pc, col in zip(parts['bodies'], colors_e):
        pc.set_facecolor(col); pc.set_alpha(0.65)
    parts['cmedians'].set_color('k'); parts['cmedians'].set_linewidth(2)
    ax.set_xticks(range(len(labels_e)))
    ax.set_xticklabels(labels_e, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', alpha=0.35, linestyle='--')
    ax.tick_params(axis='y', labelsize=8)


# ── Main ──────────────────────────────────────────────────────────────────────

def view_cluster():
    root = Tk(); root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Selecciona camera_data.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()
    if not csv_path:
        return

    n_clusters = _ask_clusters()

    df = pd.read_csv(csv_path)

    # ── Clustering por posición y altura ──────────────────────────────────────
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[['pos_x', 'pos_y', 'height']].values)
    df['cluster'] = KMeans(n_clusters=n_clusters, n_init=10,
                           random_state=42).fit_predict(scaled)

    # ── Layout: vista 3D + 3 violines + tabla ─────────────────────────────────
    fig = plt.figure(figsize=(22, 10))
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            width_ratios=[2.2, 1, 1],
                            hspace=0.6, wspace=0.38)
    ax3d      = fig.add_subplot(gs[:, 0], projection='3d')
    ax_pitch  = fig.add_subplot(gs[0, 1])
    ax_height = fig.add_subplot(gs[1, 1])
    ax_focal  = fig.add_subplot(gs[2, 1])
    ax_table  = fig.add_subplot(gs[:, 2])

    cmap_cl    = plt.get_cmap('tab20')
    cmap_pitch = plt.get_cmap('RdYlBu_r')          # azul=nadir · rojo=oblicuo
    norm_p     = mcolors.Normalize(vmin=df['pitch'].min(), vmax=df['pitch'].max())
    cl_colors  = [cmap_cl(c % 20) for c in range(n_clusters)]

    # ── Scatter: todas las cámaras, color = pitch ─────────────────────────────
    pos_raw  = df[['pos_x', 'pos_y', 'pos_z']].values
    pos_plot = _to_plot(pos_raw)

    sc = ax3d.scatter(
        pos_plot[:, 0], pos_plot[:, 1], pos_plot[:, 2],
        c=df['pitch'].values, cmap=cmap_pitch, norm=norm_p,
        s=20, alpha=0.55, depthshade=True, zorder=3
    )
    sm = plt.cm.ScalarMappable(cmap=cmap_pitch, norm=norm_p)
    sm.set_array([])
    fig.colorbar(sm, ax=ax3d, shrink=0.45, pad=0.1,
                 label='Pitch (°)', orientation='vertical')

    # ── Frustums: representante de cada cluster ────────────────────────────────
    mean_h      = float(df['height'].mean()) or 1.0
    scene_span  = float(np.max(np.ptp(pos_raw, axis=0))) or 1.0
    all_plot_v  = [pos_plot]
    legend_elems = []

    for cid in range(n_clusters):
        sub = df[df['cluster'] == cid]
        if sub.empty:
            continue
        color = cl_colors[cid]

        # Representante más cercano al centroide del cluster
        sub_sc   = scaled[df['cluster'].values == cid]
        centroid = sub_sc.mean(axis=0)
        rep_idx  = sub.index[np.argmin(np.linalg.norm(sub_sc - centroid, axis=1))]
        rep      = df.loc[rep_idx]

        pos  = np.array([rep['pos_x'], rep['pos_y'], rep['pos_z']])
        R_wc = np.array(ast.literal_eval(rep['R_world_flat'])).reshape(3, 3)
        fx   = float(rep.get('focal_x',   1000))
        fy   = float(rep.get('focal_y',   1000))
        px   = float(rep.get('principal_x', 500))
        py   = float(rep.get('principal_y', 400))
        h_rel = float(rep.get('height', mean_h))

        # Escala proporcional a la altura relativa de esta cámara
        scale = scene_span * 0.055 * (h_rel / mean_h)
        scale = np.clip(scale, scene_span * 0.02, scene_span * 0.15)

        v = _frustum_vertices(pos, R_wc, fx, fy, px, py, scale)
        all_plot_v.append(v)

        faces = [
            [v[0], v[1], v[2]], [v[0], v[2], v[3]],
            [v[0], v[3], v[4]], [v[0], v[4], v[1]],
            [v[1], v[2], v[3], v[4]],
        ]
        ax3d.add_collection3d(Poly3DCollection(
            faces, facecolors=color, alpha=0.45, edgecolors='k', linewidths=0.6
        ))

        # Flecha de dirección óptica
        pp       = _to_plot(pos)
        dir_cam  = R_wc @ np.array([0, 0, 1])
        dir_plot = np.array([dir_cam[0], -dir_cam[1], -dir_cam[2]])
        ax3d.quiver(*pp, *dir_plot, length=scale * 2.0,
                    color=color, linewidth=2.5, arrow_length_ratio=0.2)

        # Etiqueta con parámetros clave
        ax3d.text(
            pp[0], pp[1], pp[2] + scale * 1.2,
            f"G{cid}  p={rep['pitch']:.0f}°\nh={h_rel:.1f}  f={fy:.0f}",
            fontsize=7.5, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.65, edgecolor=color, pad=2)
        )

        legend_elems.append(Line2D(
            [0], [0], marker='s', color='w',
            label=f'G{cid}  (n={len(sub)})',
            markerfacecolor=color, markersize=9
        ))

    # Ajuste de límites 3D
    flat_v  = np.vstack(all_plot_v)
    mid     = np.mean(flat_v, axis=0)
    rng     = np.max(np.ptp(flat_v, axis=0)) / 1.6
    ax3d.set_xlim(mid[0]-rng, mid[0]+rng)
    ax3d.set_ylim(mid[1]-rng, mid[1]+rng)
    ax3d.set_zlim(mid[2]-rng, mid[2]+rng)
    ax3d.set_xlabel('X (Este)',    fontsize=9)
    ax3d.set_ylabel('−Y (Norte)',  fontsize=9)
    ax3d.set_zlabel('Z (Altura)',  fontsize=9)
    ax3d.set_title(
        f'{n_clusters} Clusters  ·  {len(df)} cámaras\n'
        'Puntos: todas (color = pitch)  ·  Pirámides: representante por cluster',
        fontsize=10
    )
    ax3d.legend(handles=legend_elems, title='Clusters',
                loc='upper left', fontsize=7.5, title_fontsize=8.5)

    # ── Violines de distribución ───────────────────────────────────────────────
    grp_pitch  = [df[df['cluster'] == c]['pitch'].values   for c in range(n_clusters)]
    grp_height = [df[df['cluster'] == c]['height'].values  for c in range(n_clusters)]
    grp_focal  = [df[df['cluster'] == c]['focal_y'].values for c in range(n_clusters)]
    cl_labels  = [f'G{c}' for c in range(n_clusters)]

    _violin(ax_pitch,  grp_pitch,  cl_labels, cl_colors, 'Pitch por Cluster',   'Pitch (°)')
    _violin(ax_height, grp_height, cl_labels, cl_colors, 'Altura Relativa',     'Altura (u.)')
    _violin(ax_focal,  grp_focal,  cl_labels, cl_colors, 'Longitud Focal  fy',  'focal_y (px)')

    # ── Tabla de estadísticas ─────────────────────────────────────────────────
    ax_table.axis('off')
    col_headers = ['Cluster', 'N', 'Pitch\nmedio (°)', 'Altura\nmedia', 'Focal Y\nmedia']
    rows = []
    for cid in range(n_clusters):
        sub = df[df['cluster'] == cid]
        if sub.empty:
            continue
        rows.append([
            f'G{cid}',
            str(len(sub)),
            f"{sub['pitch'].mean():.1f}",
            f"{sub['height'].mean():.2f}",
            f"{sub['focal_y'].mean():.0f}",
        ])

    tbl = ax_table.table(
        cellText=rows, colLabels=col_headers,
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c2c2c')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f4f4f4')
        cell.set_edgecolor('#cccccc')

    ax_table.set_title('Estadísticas por Cluster',
                       fontsize=10, fontweight='bold', pad=14)

    fig.suptitle('Análisis de Parámetros de Cámara — VGGT',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    view_cluster()
