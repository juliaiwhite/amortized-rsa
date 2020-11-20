# amortized-rsa
### [Learning to refer informatively by amortizing pragmatic reasoning](https://arxiv.org/abs/2006.00418)

- `shapeworld.py` generates the shapeworld dataset
- `models.py` contains code for the models listed in the paper
  - `/models/` contains pretrained models
- `train.py` file can be used to train the models with the following arguments:
  - `--dataset` specifies the dataset to train on (`shapeworld` or `colors`)
    - `--generalization` specifies the generalization type if training the model to generalize to new colors (`new_color`), combinations (`new_combo`), or shapes (`new_shape`)
  - `--vocab` flag generates a new vocab file
  - `--s0` flag trains a literal speaker
  - `--l0` flag trains a literal listener
  - `--amortized` flag trains an amortized speaker
    - `--penalty` specifies the utterance cost function (`length`)
      - `--lmbd` specifies the cost function parameter
