
import matplotlib.pyplot as plt

from skneuromsi.neural import Cuppini2017

# Model setup
model = Cuppini2017(
    neurons=180,
    tau=(3, 15, 1),
    s=0.3,
    theta=20,
    time_range=(0, 100),
    time_res=0.01,
)


# Model execution
res = model.run(
        auditory_position=100,
        visual_position=90,
        visual_duration=100,
        auditory_duration=100,
        noise=True,
        noise_level=0.40,
    )

# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()