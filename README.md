# case-embeddings
Research Case Embeddings

# TO RUN EXPERIMENT:
1. Start some ideally pytorch container (I used nvcr.io/nvidia/pytorch:26.01-py3)
2. Clone this repo and go to main directory
3. run `pip install -r requirements.txt`
4. run `pip install -e .`
5. run `python scripts/train.py --config-name experiment_2026_02_01.yaml` or any other config of choice
6. You can view tensorboard runs via `tensorboard --logdir runs --port <some_port_number>`
