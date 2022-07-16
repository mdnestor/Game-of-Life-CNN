# Conway's Game of Life as a one-layer convolutional neural network

This repository shows how to implement [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) as a 1-layer convolutional neural network (CNN) in PyTorch.

This is not to be confused with attempts to model Game of Life via machine learning as in [arXiv:2009.01398](https://arxiv.org/abs/2009.01398) or [arXiv:1809.02942](https://arxiv.org/abs/1809.02942).

## Usage

The model is implemented in the [demo notebook](demo/golcnn.ipynb).
It can also be run in [colab](https://colab.research.google.com/github/mdnestor/Game-of-Life-CNN/blob/master/demo/golcnn.ipynb).

```sh
git clone https://github.com/mdnestor/Game-of-Life-CNN.git
cd Game-of-Life-CNN
```

Load the convolutional neural network from [model.py](model.py):
```python
import torch
from model import GOLCNN
golcnn = GOLCNN()
```

Generate a random initial state with [37% initial density](https://arxiv.org/abs/1407.1006) and convert to tensor:

```python
import numpy as np
width, height = 100, 100
state = np.random.binomial(n=1, p=0.37, size=(width, height))
state = state.astype('float32')
state = torch.tensor(state)
state = state.unsqueeze(0).unsqueeze(0)
```

(Optional) Push to GPU:

```python
device = torch.device('cuda')
golcnn = golcnn.to(device)
state = state.to(device)
```

Simulate trajectory:

```python
import time
history = [state]
t0 = time.time()
for _ in range(1000):
    state = golcnn(state)
    history.append(state)
t1 = time.time()
print(f'Elapsed time: {t1 - t0} s')
```
> Elapsed time: 0.41646599769592285 s
