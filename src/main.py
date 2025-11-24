from data_utils.dataloader import CaseEmbeddingsData, get_dataloader
from data_utils.config import ExperimentConfig
from data_utils.preprocess import Preprocessor
from engine.runner import Runner
from models.our_model import CaseEmbeddingsModel
from models.pretrained import PretrainedModel
import numpy as np
import torch
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # load config
    experiment_config_path = "configs/experiments/experiment0.yaml"
    experiment_config = ExperimentConfig.parse_config_file(experiment_config_path)
    set_seed(experiment_config.seed)
    model_config = experiment_config.model_config
    data_config = experiment_config.data_config

    # preprocess and load data
    preprocessed_data = Preprocessor(data_config)
    dataset = CaseEmbeddingsData(preprocessed_data.X, preprocessed_data.Y)
    trainloader = get_dataloader(dataset, data_config)
    validateloader = get_dataloader(dataset, data_config) # ofc, this would be from a diff dataset. just writing this to have the var

    # load model
    model = CaseEmbeddingsModel(model_config)
    # model = PretrainedModel(model_config)

    # train and validate
    runner = Runner(model, trainloader, validateloader, experiment_config)
    train_losses, val_losses = runner.run_all()
    print(train_losses, val_losses) # plot these