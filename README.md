# master_dissertation
Recently, the layer-wise learning has been well developed into an alternative training schema of neural networks aims to bypass drawbacks brought by the traditional backpropagation (BP) learning. A newly error-based forward layerwise learning method, which is so-called forward progressive learning (FPL), has been used to construct the analytical framework of deep convolutional neural networks (CNNs). The FPL method is capable of more robust learning convergence, better performance and more explainable ability than the well-known stochastic gradient descent (SGD) method. Previous researches related to the FPL method only restrict on the classification task, but transfer learning abilities of these pre-trained models also need to be investigated to make them fit into other tasks. In this dissertation project, we proposed a simple object detection architecture, image pyramids and sliding windows (IPSW), to convert pre-trained models into object detectors. Through massive comparisons, it turns out that models pre-trained by the FPL method, especially those subnets in the analytical structure of CNNs, fine-tuned with our proposed IPSW achieve better
detection metrics but have less trainable parameters in the pre-training stage than those counterparts with the SGD method. Moreover, we also compared our proposed IPSW with other popular types of object detection architecture, such as R-CNN and faster R-CNN. It can be proven that our proposed IPSW is a more suitable option for the evaluation of transfer learning abilities of pretrained models with FPL method in the object detection field.

#### the environment is tensorflow-gpu==2.2.0


#### the code for the FPL method can be found in https://github.com/singularity6033/fpl


#### the code to convert pytorch weights to tensorflow weights can be found in https://github.com/singularity6033/convert_pytorch_to_tensorflow
