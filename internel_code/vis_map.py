import matplotlib.pyplot as plt
import numpy as np

data_rate = [0.1, 0.25, 0.5, 0.75, 1.0]
ap_score = [34.4, 44.5, 50.8, 54.5, 57.0]

fig = plt.figure(figsize=(10, 5))
plt.plot(data_rate, ap_score, marker='o')
plt.grid(True)
plt.xlabel('Size of Train Set (1.0 = 61659)')
plt.ylabel('mAP of VEHICLE')
fig.savefig('internal_code/ap_score.png', dpi=100)
plt.close()