import matplotlib.pyplot as plt

from skneuromsi.neural import Cuppini2017

# Model setup
model_cuppini2017 = Cuppini2017(neurons=90, 
                                position_range=(0, 90))

# Model execution
res = model_cuppini2017.run(auditory_position=35, 
                            visual_position=52)

# Results plot
ax1 = plt.subplot()
res.plot.linep(ax=ax1)
ax1.set_ylabel("neural activity")
ax1.set_xlabel("stimulus location (deg)")

plt.show()

