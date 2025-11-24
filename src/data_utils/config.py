import yaml

class DictKeysAsAttrs(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            return DictKeysAsAttrs(value)
        return value

class BasicConfig:
    @classmethod
    def parse_config_file(cls, config_path: str):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return DictKeysAsAttrs(config)


class ModelConfig(BasicConfig):
    @classmethod
    def parse_config_file(cls, config_path: str):
        return super().parse_config_file(config_path)


class DataConfig(BasicConfig):
    @classmethod
    def parse_config_file(cls, config_path: str):
        return super().parse_config_file(config_path)


class ExperimentConfig(BasicConfig):
    def __init__(self, name: str, seed: int, model_config: ModelConfig, data_config: DataConfig):
        self.name = name
        self.seed = seed
        self.model_config = model_config
        self.data_config = data_config

    @classmethod
    def parse_config_file(cls, config_path: str):
        config = super().parse_config_file(config_path)
        model_config = ModelConfig.parse_config_file(config.model_config_path)
        data_config = DataConfig.parse_config_file(config.data_config_path)
        return cls(config.name, config.seed, model_config, data_config)