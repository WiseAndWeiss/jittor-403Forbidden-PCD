from .spec import ModelSpec
from .vm import VelocityModule
from .straight_vm import StraightVelocityModule
from .straightpcf import StraightPCF


def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'StraightVelocityModule': StraightVelocityModule,
        'StraightPCF': StraightPCF,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
