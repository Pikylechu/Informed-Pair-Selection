### Abstract
Siamese Neural Networks (SNNs) are deep metric learners that use paired instance comparisons to learn similarity. The neural feature maps learnt in this way provide useful representations for classification tasks. Learning in SNNs is not reliant on explicit class knowledge; instead they require knowledge about the relationship between pairs. Though often ignored, we have found that appropriate pair selection is crucial to maximising training efficiency, particularly in scenarios where time or examples are limited. In this paper, we study the role of informed pair selection and propose a 2-phased strategy of exploration and exploitation. Random sampling provides the needed coverage for exploration, while areas of uncertainty modelled by neighbourhood properties of the pairs drive exploitation. We adopt curriculum learning to organise the ordering of pairs at training time using similarity knowledge as a heuristic for pair sorting. The results of our experimental evaluation on 4 datasets show that these strategies are key to optimising training.


### Purpose
This online code repository exists to assist in reproducibility of experiments for [ECML-PKDD 2018](http://www.ecmlpkdd2018.org/).

Each experiment has been completed using Python 2.7 and Keras version 1.2.0 - though minimum adaptation could ensure compatibility with Python 3 and Keras 2.0 or above respectively.


### Key
Each python file corresponds to one of the algorithms discussed in the paper. To consolidate the names of the algorithm, a key is provided below.

  [Base.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Base.py) = Base
  
  [Base_Ordered.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Base_Ordered.py) = Base*
  
  [Dyne.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Dyne.py) = DynE
  
  [Dyne_Ordered.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Dyne_Ordered.py) = DynE*
  
  [Dynee.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Dynee.py) = DynEE
  
  [Dynee_Ordered.py](https://github.com/Kyle-RGU/Informed-Pair-Selection/blob/master/Informed%20Pair%20Selection/SNN_Dynee_Ordered.py) = DynEE*
  

### Datasets
We have also attached here the LMRD dataset used, as well as our specific Doc2Vec model created and used to achieve the results discussed in the paper. 

Though we have not included the MNIST or SelfBACK datasets in this repository, these are easily available at the following sources:

  MNIST - Load from Keras
  
    from keras.datasets import mnist
  
  SelfBACK - https://github.com/selfback/activity-recognition
