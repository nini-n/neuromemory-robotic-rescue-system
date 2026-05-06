import os
import math
import numpy as np
import plotly.graph_objects as go

# ============================================================
# NeuroMemory Robot - Advanced 3D Rescue Scene Renderer
# Saves multiple 3D views as PNG + interactive HTML
# ============================================================

OUTPUT_DIR = "outputs/advanced_3d_views"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------
# Geometry helper functions
# ----------------------------
def add_cuboid(fig, origin, size, color, opacity=1.0, name="", showlegend=True):
    x, y, z = origin
    dx, dy, dz = size

    vertices = np.array([
        [x,     y,     z],
        [x+dx,  y,     z],
        [x+dx,  y+dy,  z],
        [x,     y+dy,  z],
        [x,     y,     z+dz],
        [x+dx,  y,     z+dz],
        [x+dx,  y+dy,  z+dz],
        [x,     y+dy,  z+dz],
    ])

    i = [0, 0, 0, 1, 1, 2, 4, 4, 5, 5, 6, 7]
    j = [1, 2, 3, 2, 5, 3, 5, 7, 6, 1, 7, 4]
    k = [2, 3, 1, 5, 6, 6, 6, 6, 1, 0, 3, 3]

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        flatshading=True,
        lighting=dict(
            ambient=0.45,
            diffuse=0.8,
            specular=0.15,
            roughness=0.85,
            fresnel=0.05,
        ),
        lightposition=dict(x=100, y=120, z=200),
        hovertemplate=name + "<extra></extra>"
    ))


def add_cylinder(fig, center, radius, height, color, axis="z", opacity=1.0, name="", showlegend=True, resolution=40):
    theta = np.linspace(0, 2*np.pi, resolution)
    h = np.linspace(0, height, 2)
    theta_grid, h_grid = np.meshgrid(theta, h)

    cx, cy, cz = center

    if axis == "z":
        x = cx + radius * np.cos(theta_grid)
        y = cy + radius * np.sin(theta_grid)
        z = cz + h_grid
    elif axis == "x":
        x = cx + h_grid
        y = cy + radius * np.cos(theta_grid)
        z = cz + radius * np.sin(theta_grid)
    elif axis == "y":
        x = cx + radius * np.cos(theta_grid)
        y = cy + h_grid
        z = cz + radius * np.sin(theta_grid)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        showscale=False,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        colorscale=[[0, color], [1, color]],
        hovertemplate=name + "<extra></extra>"
    ))

    # bottom cap
    theta2 = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, 18)
    theta2_grid, r_grid = np.meshgrid(theta2, r)

    if axis == "z":
        xb = cx + r_grid * np.cos(theta2_grid)
        yb = cy + r_grid * np.sin(theta2_grid)
        zb = np.full_like(xb, cz)

        xt = cx + r_grid * np.cos(theta2_grid)
        yt = cy + r_grid * np.sin(theta2_grid)
        zt = np.full_like(xt, cz + height)

    elif axis == "x":
        xb = np.full_like(r_grid, cx)
        yb = cy + r_grid * np.cos(theta2_grid)
        zb = cz + r_grid * np.sin(theta2_grid)

        xt = np.full_like(r_grid, cx + height)
        yt = cy + r_grid * np.cos(theta2_grid)
        zt = cz + r_grid * np.sin(theta2_grid)

    elif axis == "y":
        xb = cx + r_grid * np.cos(theta2_grid)
        yb = np.full_like(r_grid, cy)
        zb = cz + r_grid * np.sin(theta2_grid)

        xt = cx + r_grid * np.cos(theta2_grid)
        yt = np.full_like(r_grid, cy + height)
        zt = cz + r_grid * np.sin(theta2_grid)

    fig.add_trace(go.Surface(
        x=xb, y=yb, z=zb,
        showscale=False,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        showlegend=False
    ))

    fig.add_trace(go.Surface(
        x=xt, y=yt, z=zt,
        showscale=False,
        opacity=opacity,
        colorscale=[[0, color], [1, color]],
        hoverinfo="skip",
        showlegend=False
    ))


def add_sphere(fig, center, radius, color, opacity=1.0, name="", showlegend=True, resolution=30):
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    u_grid, v_grid = np.meshgrid(u, v)

    cx, cy, cz = center
    x = cx + radius * np.cos(u_grid) * np.sin(v_grid)
    y = cy + radius * np.sin(u_grid) * np.sin(v_grid)
    z = cz + radius * np.cos(v_grid)

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        showscale=False,
        opacity=opacity,
        name=name,
        showlegend=showlegend,
        colorscale=[[0, color], [1, color]],
        hovertemplate=name + "<extra></extra>"
    ))


def add_line(fig, points, color, width=8, name="", showlegend=True):
    pts = np.array(points)
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="lines+markers",
        line=dict(color=color, width=width),
        marker=dict(size=3, color=color),
        name=name,
        showlegend=showlegend,
        hovertemplate=name + "<extra></extra>"
    ))


def add_label(fig, position, text, color="white", size=11):
    x, y, z = position
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode="text",
        text=[text],
        textfont=dict(color=color, size=size),
        showlegend=False,
        hoverinfo="skip"
    ))


# ----------------------------
# Build main scene
# ----------------------------
def build_scene():
    fig = go.Figure()

    # Ground plane
    gx = np.array([[-6, 6], [-6, 6]])
    gy = np.array([[-5, -5], [5, 5]])
    gz = np.array([[0, 0], [0, 0]])

    fig.add_trace(go.Surface(
        x=gx, y=gy, z=gz,
        showscale=False,
        opacity=1.0,
        colorscale=[[0, "#2c2f36"], [1, "#3a3f48"]],
        name="Rescue ground",
        showlegend=True,
        hovertemplate="Rescue ground<extra></extra>"
    ))

    # Low visibility / smoke zone
    add_cuboid(
        fig,
        origin=(1.2, -0.6, 0.0),
        size=(3.1, 2.8, 1.8),
        color="rgba(120,130,150,0.40)",
        opacity=0.18,
        name="Low-visibility zone"
    )

    # Debris blocks
    debris_list = [
        ((-4.4,  2.2, 0.0), (1.7, 1.2, 0.8), "#7b6d5d", "Debris 1"),
        ((-3.0, -0.7, 0.0), (1.0, 1.7, 1.2), "#726556", "Debris 2"),
        (( 2.4,  2.0, 0.0), (1.5, 1.4, 0.9), "#7b6d5d", "Debris 3"),
        ((-0.8, -3.2, 0.0), (2.1, 1.1, 0.7), "#74675a", "Debris 4"),
    ]
    for origin, size, color, name in debris_list:
        add_cuboid(fig, origin, size, color, opacity=1.0, name=name, showlegend=(name == "Debris 1"))

    # Risk zone marker
    add_cylinder(fig, center=(1.55, 0.65, 0.0), radius=0.14, height=0.12,
                 color="#f4c542", opacity=1.0, name="Risk zone marker")

    # Survivor candidate
    add_cylinder(fig, center=(3.45, 1.25, 0.0), radius=0.17, height=0.72,
                 color="#d95f5f", opacity=1.0, name="Survivor candidate")

    # Rescue robot body
    add_cylinder(fig, center=(-1.8, -1.3, 0.0), radius=0.25, height=0.34,
                 color="#2e86de", opacity=1.0, name="Rescue robot body")

    # Camera mast
    add_cylinder(fig, center=(-1.8, -1.3, 0.34), radius=0.045, height=0.78,
                 color="#7f8c8d", opacity=1.0, name="Robot camera mast")

    # Sensor head
    add_cuboid(fig, origin=(-1.93, -1.43, 1.12), size=(0.26, 0.26, 0.10),
               color="#d6eaf8", opacity=1.0, name="Robot sensor head")

    # Next best view marker
    add_sphere(fig, center=(2.75, -0.10, 0.25), radius=0.12,
               color="#00d2d3", opacity=1.0, name="Next-best-view marker")

    # Standard path (short but riskier)
    standard_path = [
        (-1.8, -1.3, 0.15),
        (-0.5, -1.3, 0.15),
        ( 1.1, -0.8, 0.15),
        ( 2.4,  0.2, 0.15),
        ( 3.3,  1.1, 0.15),
    ]
    add_line(fig, standard_path, color="#ff6b6b", width=10, name="Standard path")

    # Risk-aware path
    risk_aware_path = [
        (-1.8, -1.3, 0.16),
        (-0.4, -1.3, 0.16),
        ( 0.9, -1.3, 0.16),
        ( 1.4, -2.1, 0.16),
        ( 3.3, -2.1, 0.16),
        ( 4.1, -0.6, 0.16),
        ( 3.5,  1.0, 0.16),
    ]
    add_line(fig, risk_aware_path, color="#00c2a8", width=10, name="Risk-aware path")

    # Robot to NBV link
    nbv_path = [
        (-1.8, -1.3, 0.22),
        (-0.5, -1.3, 0.22),
        ( 1.0, -1.0, 0.22),
        ( 2.75, -0.10, 0.25),
    ]
    add_line(fig, nbv_path, color="#4dabf7", width=6, name="Re-observation path")

    # Labels
    add_label(fig, (-2.1, -1.7, 0.45), "Rescue robot", color="#a9d6ff", size=12)
    add_label(fig, (3.55, 1.30, 0.95), "Survivor candidate", color="#ffb0b0", size=12)
    add_label(fig, (2.85, -0.10, 0.48), "Next-best view", color="#9ff8f8", size=11)
    add_label(fig, (2.45, 1.65, 1.95), "Low-visibility zone", color="#d5dbe7", size=12)
    add_label(fig, (1.7, 0.7, 0.35), "Risk zone", color="#ffe08a", size=11)

    # Layout
    fig.update_layout(
        title=dict(
            text="NeuroMemory Robot - 3D Rescue Scene",
            x=0.5,
            font=dict(size=24)
        ),
        template="plotly_dark",
        width=1600,
        height=1000,
        margin=dict(l=10, r=10, b=10, t=55),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(0,0,0,0.35)",
            borderwidth=0
        ),
        scene=dict(
            xaxis=dict(
                visible=False,
                range=[-6, 6],
                backgroundcolor="rgb(22,22,22)"
            ),
            yaxis=dict(
                visible=False,
                range=[-5, 5],
                backgroundcolor="rgb(22,22,22)"
            ),
            zaxis=dict(
                visible=False,
                range=[0, 3.2],
                backgroundcolor="rgb(22,22,22)"
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.6, y=1.3, z=0.65),
            bgcolor="rgb(8,10,14)",
        )
    )

    return fig


# ----------------------------
# Save multiple view angles
# ----------------------------
def save_views(fig):
    view_dict = {
        "view_01_overview": dict(
            eye=dict(x=1.75, y=1.55, z=1.10),
            center=dict(x=0.0, y=0.0, z=-0.08)
        ),
        "view_02_top_down": dict(
            eye=dict(x=0.01, y=0.01, z=3.1),
            center=dict(x=0.0, y=0.0, z=0.0)
        ),
        "view_03_side_robot_focus": dict(
            eye=dict(x=-2.2, y=-2.8, z=1.0),
            center=dict(x=-0.05, y=0.05, z=-0.05)
        ),
        "view_04_survivor_focus": dict(
            eye=dict(x=2.5, y=2.2, z=0.9),
            center=dict(x=0.35, y=0.05, z=-0.05)
        ),
        "view_05_planning_angle": dict(
            eye=dict(x=0.2, y=-3.2, z=1.4),
            center=dict(x=0.12, y=-0.05, z=-0.05)
        ),
    }

    for name, camera in view_dict.items():
        fig.update_layout(scene_camera=camera)
        out_path = os.path.join(OUTPUT_DIR, f"{name}.png")
        fig.write_image(out_path, scale=2)
        print(f"Saved: {out_path}")


def save_html(fig):
    html_path = os.path.join(OUTPUT_DIR, "neuromemory_3d_scene_interactive.html")
    fig.write_html(html_path)
    print(f"Saved: {html_path}")


def main():
    fig = build_scene()
    save_views(fig)
    save_html(fig)
    print("\nAll 3D scene renders created successfully.")
    print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()