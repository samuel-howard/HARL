"""Algorithm registry."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hatrpo import HATRPO
from harl.algorithms.actors.haa2c import HAA2C
from harl.algorithms.actors.haddpg import HADDPG
from harl.algorithms.actors.hatd3 import HATD3
from harl.algorithms.actors.had3qn import HAD3QN
from harl.algorithms.actors.maddpg import MADDPG
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.happo_sr import HAPPO_SR

ALGO_REGISTRY = {
    "happo": HAPPO,
    "hatrpo": HATRPO,
    "haa2c": HAA2C,
    "haddpg": HADDPG,
    "hatd3": HATD3,
    "had3qn": HAD3QN,
    "maddpg": MADDPG,
    "mappo": MAPPO,
    "happo_sr": HAPPO_SR,
}