from pathlib import Path

import yaml


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"config key {name!r} not found") from None

    def __setattr__(self, name, value):
        self[name] = value

    def to_dict(self):
        return dict(self)


def load_config(path):
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return Config(data)
