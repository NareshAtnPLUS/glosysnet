# Glosysnet

This is a package that contains all mathematical tools requied for deep learning.
Current version many NeuralNet Functions
1.Activation functions
* Sigmoid
* tanh
* relu
* leaky_relu
* elu 

			2.Classifier functions
			   - SoftMax
			3.Loss functions
			   - Mean Square
			   - Mean Absolute
			   - Mean Bias
			   - Logistic loss or negative log likelihood
			4. Optimization Functions
			   - Stohastic Gradient Descent
			   - Adagrad
			   - RMSProp
and vision functions 
			1.Convoluion
			   - conv2D
			   - maxpooling
##	Building the package from source code
run this command to build package
```bash
python setup.py sdist bdist_wheel
```
## Installing the built package
Commmand to install the package
```bash
cd dist
pip install glosysnet-version-py3-none-any.whl
```
## Import the package into your code.
```python
import glosysnet as gl
```


