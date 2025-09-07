import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Simulasi model TD3
class DummyTD3:
    def predict(self, state):
        # Action output: 6 nilai -1 sampai 1
        action = np.random.uniform(-1, 1, size=6)
        return action


# Inisialisasi model dan state
model = DummyTD3()

fig, ax = plt.subplots()
action = model.predict(np.random.rand(6))
# Ubah action (1D) menjadi 2D grid untuk visualisasi, misal 2x3
grid_shape = (2, 3)
action_grid = action.reshape(grid_shape)

im = ax.imshow(action_grid, cmap="coolwarm", vmin=-1, vmax=1)


def update(frame):
    # Simulasi state: nilai 0-1 sebanyak 6 dimensi
    state = np.random.rand(6)
    # Prediksi aksi dari model
    action = model.predict(state)
    action_grid = action.reshape(grid_shape)
    im.set_array(action_grid)
    return [im]


ani = animation.FuncAnimation(fig, update, interval=500, blit=True)

plt.title("TD3 Action Heatmap (Simulasi)")
plt.colorbar(im, ax=ax, label="Action (-1 to 1)")
plt.show()
