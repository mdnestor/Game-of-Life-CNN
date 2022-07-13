# Conway's Game of Life as a one-layer convolutional neural network

This repository shows how to implement [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) as a convolutional neural network (CNN) in PyTorch.
This is not to be confused with attempts to model Game of Life via machine learning as in [1], [2] - rather, it shows that the Game of Life can *literally* be represented as a 1-layer CNN with a specific convolution kernel and bump activation function.

This project is inspired by [neuralpatterns.io](https://github.com/MaxRobinsonTheGreat/NeuralPatterns), a web application for simulating so-called neural cellular automata and also [Lenia](https://chakazul.github.io/lenia.html) ([3]), a "Life"-life reaction-diffusion system built on alternating convolution and pointwise activation.

## Setup

The main module is implemented in [model.py](model.py), and the [demo notebook](demo/golcnn.ipynb) shows how to run a simulation from a given initial configuation, as well as how to run the simulation on a GPU for mega speed!

It can also be run in [Google Colab](https://colab.research.google.com/github/mdnestor/Game-of-Life-CNN/blob/master/demo/golcnn.ipynb).

## References

[1]: Jacob M. Springer, Garrett T. Kenyon. "It's Hard for Neural Networks To Learn the Game of Life".  2020. [arXiv:2009.01398](https://arxiv.org/abs/2009.01398)

[2]: William Gilpin. "Cellular automata as convolutional neural networks". 2020. [arXiv:1809.02942](https://arxiv.org/abs/1809.02942)

[3]: Bert Wang-Chak Chan. "Lenia - Biology of Artificial Life". [arXiv:1812.05433](https://arxiv.org/abs/1812.05433)

![output](demo/output2.gif)


