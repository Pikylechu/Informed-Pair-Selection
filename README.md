Informed Pair Selection for Self-Paced Metric Learning in Siamese Neural Networks

Abstract:
Siamese Neural Networks (SNNs) are deep metric learners that use paired instance comparisons to learn similarity. The neural feature maps learnt in this way provide useful representations for classification tasks. Learning in SNNs is not reliant on explicit class knowledge; instead they require knowledge about the relationship between pairs. Though often ignored, we have found that appropriate pair selection is crucial to maximising training efficiency, particularly in scenarios where time or examples are limited. In this paper, we study the role of informed pair selection and propose a 2-phased strategy of exploration and exploitation. Random sampling provides the needed coverage for exploration, while areas of uncertainty modelled by neighbourhood properties of the pairs drive exploitation. We adopt curriculum learning to organise the ordering of pairs at training time using similarity knowledge as a heuristic for pair sorting. The results of our experimental evaluation on 4 datasets show that these strategies are key to optimising training.


This online code repository exists to assist in reproducibility of experiments for PKDD-ECML 2018.

Each experiment has been completed using Python 2.7 and Keras version 1.2.0 - though minimum adaptation could ensure compatibility with Python 3 and Keras 2.0 or above respectively.

Each python file corresponds to an algorithm discussed in the paper. To consolidate the names of the algorithm, a key is provided below.

  Base = Base
  
  Base_Ordered = Base*
  
  Dyne = DynE
  
  Dyne_Ordered = DynE*
  
  Dynee = DynEE
  
  Dynee_Ordered = DynEE*

We have also attached here the LMRD dataset used, as well as our specific Doc2Vec model created and used to achieve the results discussed in the paper. 

Though we have not included the MNIST or SelfBACK datasets in this repository, these are easily available at the following sources:
  MNIST - Load from Keras
  SelfBACK - https://github.com/selfback/activity-recognition
