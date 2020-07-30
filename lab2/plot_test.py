import numpy as np
import matplotlib.pyplot as plt

# Make a fake dataset
height = [3, 12, 5, 18, 45]
bars = ('Az', 'B', 'C', 'D', 'E')
y_pos = np.arange(len(bars))

plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
plt.xticks(y_pos, bars)
plt.show()
