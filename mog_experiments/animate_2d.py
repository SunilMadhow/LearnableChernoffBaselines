from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def animate_traj(traj, rewards=None, outfile="trajectory.gif", interval=200, cmap="viridis"):
    """
    Animate a trajectory array of shape (T, S, 2).
    Renders a scatter of all S points per timestep and saves to outfile.

    rewards: optional array of shape (S,) used to color points by final reward.
    """
    traj = np.asarray(traj)
    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(f"Expected traj shape (T, S, 2), got {traj.shape}")

    T, S, _ = traj.shape
    x_min, y_min = traj.min(axis=(0, 1)) - 0.5
    x_max, y_max = traj.max(axis=(0, 1)) + 0.5

    colors = None
    norm = None
    if rewards is not None:
        rewards = np.asarray(rewards)
        if rewards.shape != (S,):
            raise ValueError(f"Expected rewards shape {(S,)}, got {rewards.shape}")
        norm = plt.Normalize(vmin=rewards.min(), vmax=rewards.max())
        colors = rewards

    fig, ax = plt.subplots()
    scat = ax.scatter(traj[0, :, 0], traj[0, :, 1], s=10, alpha=0.7, c=colors, cmap=cmap, norm=norm)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Trajectory over time")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if rewards is not None:
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label("Reward")

    def init():
        scat.set_offsets(traj[0])
        ax.set_title(f"Trajectory at t=0 / {T-1}")
        return scat,

    def update(frame):
        scat.set_offsets(traj[frame])
        ax.set_title(f"Trajectory at t={frame} / {T-1}")
        return scat,

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        blit=True,
        interval=interval,
        repeat=False,
    )
    anim.save(outfile, writer="pillow")
    plt.close(fig)
    return outfile


def animate_val_predictions(states, index_set, model, outfile="val_predictions.gif",
                            interval=200, cmap="viridis", steps=50, padding=0.5):
    """
    Animate value-net predictions over time on a grid of points.
    states: array-like, shape (T, S, 2) used to set plot bounds.
    model: ValueEstimate instance
    """
    traj = np.asarray(states)
    if traj.ndim != 3 or traj.shape[2] != 2:
        raise ValueError(f"Expected states shape (T, S, 2), got {traj.shape}")
    index_set = np.asarray(index_set)
    if index_set.shape[0] != traj.shape[0]:
        raise ValueError(f"index_set length {index_set.shape[0]} does not match T={traj.shape[0]}")

    T, _, _ = traj.shape
    x_min, y_min = traj.min(axis=(0, 1)) - padding
    x_max, y_max = traj.max(axis=(0, 1)) + padding
    xs = np.linspace(x_min, x_max, steps)
    ys = np.linspace(y_min, y_max, steps)
    Xg, Yg = np.meshgrid(xs, ys)
    grid_pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)

    preds_per_t = []
    for t in range(T):
        tt = np.full(len(grid_pts), index_set[t], dtype=np.float32)
        preds = model.predict_under_log(grid_pts, t)
        preds_per_t.append(preds)
    preds_per_t = np.stack(preds_per_t, axis=0)
    print("Predictions shape:", preds_per_t.shape)
    vmin, vmax = preds_per_t.min(), preds_per_t.max()
    print("Value prediction range:", vmin, vmax)

    fig, ax = plt.subplots()
    cf = ax.contourf(Xg, Yg, preds_per_t[0].reshape(Xg.shape), levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Value predictions over time")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label("Predicted value")

    def init():
        cf = ax.contourf(Xg, Yg, preds_per_t[0].reshape(Xg.shape), levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        return cf.collections

    def update(frame):
        for coll in ax.collections:
            coll.remove()
        cf = ax.contourf(Xg, Yg, preds_per_t[frame].reshape(Xg.shape), levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"t={frame} / {T-1}")
        return cf.collections

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        blit=True,
        interval=interval,
        repeat=False,
    )
    anim.save(outfile, writer="pillow")
    plt.close(fig)
    return outfile
