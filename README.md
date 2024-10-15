# HIV-AICare

This is the code for the HIV-AICare. To run it, use the following line:

## Parameters

| Parameter           | Type     | Default              | Description                                                  |
|---------------------|----------|----------------------|--------------------------------------------------------------|
| `--input`           | `str`    | N/A                  | Path to the input file or dataset.                           |
| `--model`           | `str`    | `model.pth`          | Path to the pre-trained model file.                          |
| `--output`          | `str`    | `results/`           | Directory where the output will be saved.                    |
| `--epochs`          | `int`    | `10`                 | Number of training epochs.                                   |
| `--batch-size`      | `int`    | `32`                 | Number of samples per batch during training.                 |
| `--learning-rate`   | `float`  | `0.001`              | Learning rate for the optimizer.                             |
| `--use-gpu`         | `bool`   | `False`              | Set to `True` to use GPU for computations.                   |


python main.py  --seed $seed --epochs $epochs --n $n --eta $eta --lr $lr --n_interact_past_state 3  --n_action 50 --embed_dim $embed --n_timestep 10 --eval_episodes 1  --lstm True --implicit_q True --logdir $logdir  --hidden 64 --binary_vector True
