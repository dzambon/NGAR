# Autoregressive Models for Sequences of Graphs

This is the official implementation of:

**Autoregressive Models for Sequences of Graphs.**  
[Daniele Zambon\*](https://github.com/dan-zam), [Daniele Grattarola\*](https://github.com/danielegrattarola), Lorenzo Livi, Cesare Alippi.  
[https://arxiv.org/abs/1903.07299](https://arxiv.org/abs/1903.07299)  
International Joint Conference on Neural Networks (2019).  

\* Equal contribution

Please cite the paper if you use any of this code for your own research: 

```
@article{zambon2019autoregressive,
  title={Autoregressive Models for Sequences of Graphs},
  author={Zambon, Daniele and Grattarola, Daniele and Livi, Lorenzo and Alippi, Cesare},
  journal={International Joint Conference on Neural Networks},
  year={2019}
}
```

## Abstract

This paper proposes an autoregressive (AR) model for sequences of graphs, which generalises traditional AR models. A first novelty consists in formalising the AR model for a very general family of graphs, characterised by a variable topology, and attributes associated with nodes and edges. A graph neural network (GNN) is also proposed to learn the AR function associated with the graph-generating process (GGP), and subsequently predict the next graph in a sequence. The proposed method is compared with four baselines on synthetic GGPs, denoting a significantly better performance on all considered problems. 

## Setup

The code is implemented in Python 3.5+ and was tested on Ubuntu 16.04.  
The following libraries are required to run the code : 

- [Keras](https://keras.io/), a high-level API for deep learning;
- [Spektral](https://danielegrattarola.github.io/spektral/), a library for building graph neural networks with Keras.

Both libraries are available throug PyPi: 

```bash
pip install keras
pip install spektral
```

## Running experiments

The `src` folder includes all the necessary code to reproduce the results of the paper.   
To test the GNN and baselines, simply run:

```bash
$ python src/main_gar.py
```

There is a section at the top of the script to configure hyperparameters and other experimental details. Check out the comments in the source code for more information. 
