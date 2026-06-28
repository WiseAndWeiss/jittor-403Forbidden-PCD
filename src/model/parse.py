from .spec import ModelSpec
from .vm import VelocityModule
from .asdn import ClassifyModule, ASDNModule
from .iterpfn import IterativePFNModule

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'ClassifyModule': ClassifyModule,
        'ASDNModule': ASDNModule,
        'IterativePFNModule': IterativePFNModule,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    keys = ','.join(MAP.keys())
    assert __target__ in MAP, f'expect: [{keys}], found: {__target__}'
    return MAP[__target__](model_config=model_config, **kwargs)
