import matplotlib.pyplot as plt

from skneuromsi.neural import Cuppini2017

# Model setup
model = Cuppini2017(neurons=90, position_range=(0, 90))

# Model execution
res = model.run(
    auditory_position=20,
    visual_position=65,
    auditory_sigma=32,
    visual_sigma=4,
    auditory_intensity=28,
    visual_intensity=27,
    noise=False,
    causes_kind="count",
    causes_dim="space",
    causes_peak_threshold=0.15,
    causes_peak_distance=10,
)

# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()
