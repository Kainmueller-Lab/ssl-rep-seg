BACKBONES = {}

def register(name):
    def wrapper(fn):
        BACKBONES[name] = fn
        return fn
    return wrapper


def get_backbone(name, **kwargs):
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}")
    return BACKBONES[name](**kwargs)