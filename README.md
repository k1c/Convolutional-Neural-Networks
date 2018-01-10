# COMP 4107 Convolutional-Neural-Networks

Carolyne Pelletier

Akhil Dalal

Modifying Convolutional Neural Networks (CNN) for CIFAR-10 Dataset

r = ReLU
d = Dropout
s = Softmax

CNN-1: In -> Conv (r) -> Pool (d) -> Flatten -> FC (r) -> FC (r) -> Out (s)

	To run: Python3 cnn-1.py
  
CNN-2: In -> Conv (r) -> Pool (d) -> Conv (r) -> Pool (d) -> Flatten -> FC (r) -> FC (r) -> Out (s)

	To run: Python3 cnn-2.py
  
CNN-3: In -> Conv (r) -> Pool (d) -> Conv (r) -> Pool (d) -> Conv (r) -> Flatten -> FC (r) -> FC (r) -> Out (s)

	To run: Python3 cnn-3.py
  
CNN-4: In -> {Conv (r) -> Conv (r) - > Pool (d)}*3 -> Flatten -> FC (r) -> FC (r) -> Out (s)

	To run: Python3 cnn-4.py
  

