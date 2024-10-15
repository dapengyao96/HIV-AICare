# HIV-AICare

This is the code for the HIV-AICare. To run it, use the following line:

## Parameters

| Parameter     | Type   | Default  | Description                                                    |
|---------------|--------|----------|----------------------------------------------------------------|
| `--seed`      | `int`  | `1`      | Seed for random number generation to ensure reproducibility.    |
| `--epochs`    | `int`  | `10`     | Number of epochs for training.                                 |
| `--n`         | `int`  | `10`     | The size of the data or the number of units (interpret based on your use case). |
| `--lr`        | `float`| `0.01`   | Learning rate for the optimizer. Affects the step size in weight updates during training. |



python main.py  --seed 1 --epochs 10 --n 10 --eta 0.5 --lr 0.01 --n_interact_past_state 3  --n_action 50 --embed_dim 23 --n_timestep 10 --eval_episodes 1  --lstm True --implicit_q True --logdir log  --hidden 64 --binary_vector True
