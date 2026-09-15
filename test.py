from skneuromsi.neural import Cuppini2017

cerebro = Cuppini2017(
    neurons=180, tau=(3, 15, 1), s=0.3, theta=20, seed=42,
    time_range=(0, 100), time_res=0.01,
)

res = cerebro.run(
    visual_position=90, visual_intensity=27, visual_sigma=4,
    visual_onset=0, visual_duration=100,
    auditory_position=100, auditory_intensity=28, auditory_sigma=32,
    auditory_onset=0, auditory_duration=100,
    cross_modal_weight=1.4, feedforward_weight=18,
    noise=True, noise_level=0.40,
)

print(res.causes_)
# 1) El array completo (times x positions x pos_coords), por modo,
#    como xarray.DataArray -> es lo que hay "adentro" de res
xa = res.to_xarray()
print(xa)

# 4) Todo como dict (incluye nddata, causas, parámetros de la corrida, etc.)
#d = res.to_dict()
#print(d.keys())

