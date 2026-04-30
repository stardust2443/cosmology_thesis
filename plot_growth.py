import numpy as np
import matplotlib.pyplot as plt

# Load the C++ output
z, D, f = np.loadtxt("growth_history.txt", unpack=True)

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Growth Factor D(z)
color = 'tab:blue'
ax1.set_xlabel('Redshift $z$')
ax1.set_ylabel('Linear Growth Factor $D(z)$', color=color)
ax1.plot(z, D, color=color, lw=2, label='$D(z)$')
ax1.tick_params(axis='y', labelcolor=color)

# Plot Growth Rate f(z) on secondary axis
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Growth Rate $f(z)$', color=color)  
ax2.plot(z, f, color=color, lw=2, linestyle='--', label='$f(z)$')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Cosmological Growth History ($\Lambda$CDM)")
fig.tight_layout()
plt.savefig("growth_plot.png", dpi=300)
print("Plot saved as growth_plot.png")