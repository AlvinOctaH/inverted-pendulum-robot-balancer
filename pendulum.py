import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.animation as animation

os.makedirs('assets', exist_ok=True)

# ─────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────
g  = 9.81
M  = 0.5
m  = 0.2
L  = 0.15
dt = 0.01
T  = 600   # 6 seconds

CART_W    = 0.20
CART_H    = 0.08
WHEEL_R   = 0.04
NOISE_STD = 0.005

# ─────────────────────────────────────────────────────
# CART-POLE DYNAMICS
# state = [x, x_dot, theta, theta_dot]
# ─────────────────────────────────────────────────────
def step(state, F, noise=False):
    x, xd, th, thd = state
    s, c = np.sin(th), np.cos(th)
    den  = M + m * s**2
    xdd  = (F + m*s*(L*thd**2 + g*c)) / den
    thdd = -(F*c + m*L*thd**2*s*c + (M+m)*g*s) / (L * den)
    xd2  = xd  + xdd  * dt
    thd2 = thd + thdd * dt
    x2   = x   + xd2  * dt
    th2  = th  + thd2 * dt
    if noise:
        th2 += np.random.normal(0, NOISE_STD)
    return np.array([x2, xd2, th2, thd2])

# ─────────────────────────────────────────────────────
# FULL STATE CONTROLLER
# Controls both angle (theta) AND cart position (x)
# F = Kp*th + Kd*th_dot - Kp_x*x - Kd_x*x_dot
# ─────────────────────────────────────────────────────
def controller(state, integ, prev_th, Kp, Ki, Kd, Kp_x, Kd_x):
    x, xd, th, thd = state

    # Angle term (stabilize pole)
    integ += th * dt
    deriv  = (th - prev_th) / dt
    F_th   = Kp * th + Ki * integ + Kd * deriv

    # Position term (return cart to center)
    F_x    = -Kp_x * x - Kd_x * xd

    F = np.clip(F_th + F_x, -500, 500)
    return F, integ, th

# ─────────────────────────────────────────────────────
# COST — ISE on angle + position
# ─────────────────────────────────────────────────────
def cost(params):
    Kp, Ki, Kd, Kp_x, Kd_x = params
    s = np.array([0., 0., np.pi/6, 0.])
    ig, pe, c = 0., 0., 0.
    for _ in range(T):
        F, ig, pe = controller(s, ig, pe, Kp, Ki, Kd, Kp_x, Kd_x)
        s = step(s, F)
        c += s[2]**2 + 0.01 * s[0]**2   # penalize angle + cart drift
        if abs(s[2]) > 1.2:
            return c + 5000
    return c

# ─────────────────────────────────────────────────────
# ABC OPTIMIZATION — 5 params: Kp, Ki, Kd, Kp_x, Kd_x
# ─────────────────────────────────────────────────────
def abc(n_iter=50, n_col=30):
    np.random.seed(42)
    LB = np.array([20.,  0.,  1.,  1.,  1.])
    UB = np.array([300., 5., 20., 50., 20.])

    pop = LB + np.random.rand(n_col, 5) * (UB - LB)
    fit = np.array([cost(p) for p in pop])
    hist = []

    for it in range(n_iter):
        # Employee bees
        for i in range(n_col):
            k = np.random.choice([x for x in range(n_col) if x != i])
            j = np.random.randint(5)
            c2 = pop[i].copy()
            c2[j] = np.clip(
                pop[i,j] + np.random.uniform(-1,1)*(pop[i,j]-pop[k,j]),
                LB[j], UB[j])
            f2 = cost(c2)
            if f2 < fit[i]:
                pop[i], fit[i] = c2, f2

        # Onlooker bees
        prob = (1/(1+fit)) / np.sum(1/(1+fit))
        for i in range(n_col):
            if np.random.rand() < prob[i]:
                k = np.random.choice([x for x in range(n_col) if x != i])
                j = np.random.randint(5)
                c2 = pop[i].copy()
                c2[j] = np.clip(
                    pop[i,j] + np.random.uniform(-1,1)*(pop[i,j]-pop[k,j]),
                    LB[j], UB[j])
                f2 = cost(c2)
                if f2 < fit[i]:
                    pop[i], fit[i] = c2, f2

        # Scout bees
        for i in range(n_col):
            if np.random.rand() < 0.05:
                pop[i] = LB + np.random.rand(5)*(UB-LB)
                fit[i] = cost(pop[i])

        best = pop[np.argmin(fit)]
        hist.append(np.min(fit))
        print(f"  Iter {it+1:3d}/{n_iter}  ISE={hist[-1]:.4f}  "
              f"Kp={best[0]:.1f} Ki={best[1]:.2f} Kd={best[2]:.2f} "
              f"Kp_x={best[3]:.1f} Kd_x={best[4]:.2f}")

    return pop[np.argmin(fit)], hist

# ─────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────
print("="*60)
print("  ABC Optimization — Cart-Pole Full State Control")
print("="*60)
best, hist = abc()
Kp, Ki, Kd, Kp_x, Kd_x = best
print(f"\n  Best gains:")
print(f"    Kp={Kp:.4f}  Ki={Ki:.4f}  Kd={Kd:.4f}")
print(f"    Kp_x={Kp_x:.4f}  Kd_x={Kd_x:.4f}")
print(f"  Final ISE = {hist[-1]:.6f}\n")

# Manual baseline
Kp_m, Ki_m, Kd_m, Kp_xm, Kd_xm = 50., 0., 5., 5., 2.

# ─────────────────────────────────────────────────────
# SIMULATE 4 SCENARIOS
# ─────────────────────────────────────────────────────
def simulate(Kp, Ki, Kd, Kp_x, Kd_x, noise=False):
    s = np.array([0., 0., np.pi/6, 0.])
    ig, pe = 0., 0.
    xs, ths, Fs = [], [], []
    for _ in range(T):
        F, ig, pe = controller(s, ig, pe, Kp, Ki, Kd, Kp_x, Kd_x)
        s = step(s, F, noise)
        xs.append(s[0]); ths.append(np.degrees(s[2])); Fs.append(F)
        if abs(s[2]) > 1.5:
            xs  += [xs[-1]]  * (T - len(xs))
            ths += [ths[-1]] * (T - len(ths))
            Fs  += [0]       * (T - len(Fs))
            break
    return np.array(xs[:T]), np.array(ths[:T]), np.array(Fs[:T])

x_abc, th_abc, F_abc = simulate(Kp,   Ki,   Kd,   Kp_x,  Kd_x)
x_man, th_man, F_man = simulate(Kp_m, Ki_m, Kd_m, Kp_xm, Kd_xm)
x_noi, th_noi, _     = simulate(Kp,   Ki,   Kd,   Kp_x,  Kd_x, noise=True)
_,     th_unc, _     = simulate(0,    0,    0,    0,     0)
t = np.arange(T) * dt

# ─────────────────────────────────────────────────────
# FIGURE 1: Results dashboard
# ─────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 11))
fig.suptitle('Cart-Pole Inverted Pendulum — ABC-Optimized Full State Control',
             fontsize=13, fontweight='bold')

c_abc = '#16a34a'; c_man = '#f59e0b'; c_noi = '#6366f1'; c_unc = '#ef4444'

ax = axes[0]
ax.plot(t, th_unc, '--', color=c_unc, lw=1.5, alpha=0.7, label='Uncontrolled')
ax.plot(t, th_man, '-',  color=c_man, lw=1.8,
        label=f'Manual  (Kp={Kp_m}, Kd={Kd_m}, Kp_x={Kp_xm})')
ax.plot(t, th_abc, '-',  color=c_abc, lw=2.2,
        label=f'ABC  (Kp={Kp:.1f}, Kd={Kd:.2f}, Kp_x={Kp_x:.1f})')
ax.plot(t, th_noi, ':',  color=c_noi, lw=1.5, alpha=0.9,
        label='ABC + sensor noise (σ=0.005 rad)')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_ylabel('Angle (°)'); ax.set_title('Pendulum Angle Response')
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlim(0, T*dt)

ax = axes[1]
ax.plot(t, x_man, '-', color=c_man, lw=1.8, label='Manual')
ax.plot(t, x_abc, '-', color=c_abc, lw=2.2, label='ABC')
ax.plot(t, x_noi, ':', color=c_noi, lw=1.5, alpha=0.9, label='ABC + noise')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_ylabel('Cart Position (m)'); ax.set_title('Cart Position Along Track')
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlim(0, T*dt)

ax = axes[2]
ax.plot(t, F_man, '-', color=c_man, lw=1.8, label='Manual')
ax.plot(t, F_abc, '-', color=c_abc, lw=2.2, label='ABC')
ax.axhline(0, color='k', lw=0.7, ls=':')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Force (N)')
ax.set_title('Control Force Output')
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_xlim(0, T*dt)

plt.tight_layout()
plt.savefig('assets/results_dashboard.png', dpi=150, bbox_inches='tight')
print("Saved → assets/results_dashboard.png")
plt.show()

# ─────────────────────────────────────────────────────
# FIGURE 2: Convergence
# ─────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(range(1, len(hist)+1), hist, color='#2563eb', lw=2)
ax2.fill_between(range(1, len(hist)+1), hist, alpha=0.1, color='#2563eb')
ax2.set_xlabel('Iteration'); ax2.set_ylabel('Best ISE')
ax2.set_title('ABC Optimization Convergence')
ax2.grid(alpha=0.25)
ax2.annotate(
    f'Final ISE={hist[-1]:.4f}\nKp={Kp:.1f} Kd={Kd:.2f} Kp_x={Kp_x:.1f}',
    xy=(len(hist), hist[-1]),
    xytext=(len(hist)*0.45, (hist[0]+hist[-1])/2),
    arrowprops=dict(arrowstyle='->', color='#64748b'), fontsize=9)
plt.tight_layout()
plt.savefig('assets/convergence.png', dpi=150, bbox_inches='tight')
print("Saved → assets/convergence.png")
plt.show()

# ─────────────────────────────────────────────────────
# GIF: Cart-pole animation with position control
# ─────────────────────────────────────────────────────
print("\nGenerating simulation GIF (this may take ~30 seconds)...")

s = np.array([0., 0., np.pi/6, 0.])
ig, pe = 0., 0.
anim_frames = []
for _ in range(200):
    F, ig, pe = controller(s, ig, pe, Kp, Ki, Kd, Kp_x, Kd_x)
    s = step(s, F)
    anim_frames.append((float(s[0]), float(s[2]), float(F)))

fig3, ax3 = plt.subplots(figsize=(8, 4))
TRK = 1.0

def draw(i):
    ax3.clear()
    cx, th, F = anim_frames[i]
    ax3.set_xlim(-TRK/2, TRK/2); ax3.set_ylim(-0.12, 0.52)
    ax3.set_facecolor('#f8fafc'); ax3.set_aspect('equal')

    # Track
    ax3.plot([-TRK/2, TRK/2], [0, 0], color='#475569', lw=3)
    for xd in np.arange(-TRK/2, TRK/2, 0.1):
        ax3.plot([xd, xd+0.05], [-0.005, -0.005], color='#94a3b8', lw=1)

    # Center marker
    ax3.plot([0, 0], [-0.01, 0.01], color='#ef4444', lw=2)

    # Wheels
    for wx in [cx - CART_W/2 + WHEEL_R + 0.01,
               cx + CART_W/2 - WHEEL_R - 0.01]:
        ax3.add_patch(Circle((wx, WHEEL_R), WHEEL_R,
                              color='#1e293b', zorder=3))
        ax3.add_patch(Circle((wx, WHEEL_R), WHEEL_R*0.45,
                              color='#64748b', zorder=4))

    # Cart body
    ax3.add_patch(Rectangle(
        (cx - CART_W/2, WHEEL_R*2), CART_W, CART_H,
        facecolor='#3b82f6', edgecolor='#1d4ed8',
        linewidth=1.5, zorder=2))

    # Pole
    bx, by = cx, WHEEL_R*2 + CART_H
    tx = bx + L*2*np.sin(th)
    ty = by + L*2*np.cos(th)
    ax3.plot([bx,tx],[by,ty], color='#dc2626', lw=6,
             solid_capstyle='round', zorder=5)
    ax3.plot([bx,tx],[by,ty], color='#fca5a5', lw=2,
             solid_capstyle='round', zorder=6)
    ax3.add_patch(Circle((tx, ty), 0.022, color='#7f1d1d', zorder=7))

    # Force arrow
    if abs(F) > 2:
        al = np.clip(F/200*0.15, -0.15, 0.15)
        ax3.annotate('',
            xy=(cx+al, WHEEL_R*2+CART_H/2),
            xytext=(cx, WHEEL_R*2+CART_H/2),
            arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2.5))

    ax3.set_title(
        f'θ = {np.degrees(th):+.2f}°   x = {cx:+.3f} m   F = {F:+.1f} N',
        fontsize=9.5, color='#1e293b')
    ax3.set_xlabel(
        f'ABC  Kp={Kp:.1f}  Kd={Kd:.2f}  Kp_x={Kp_x:.1f}  Kd_x={Kd_x:.2f}',
        fontsize=8, color='#64748b')
    ax3.grid(alpha=0.2)

anim = animation.FuncAnimation(fig3, draw,
                                frames=len(anim_frames), interval=10)
anim.save('assets/simulation.gif', writer='pillow', fps=30, dpi=110)
print("Saved → assets/simulation.gif")
plt.show()

print("\n" + "="*55)
print("  All done! Files saved to assets/:")
print("    results_dashboard.png")
print("    convergence.png")
print("    simulation.gif")
print("="*55)