The following is my github repo for My CodeThrough journey.
https://github.com/gwynbleiddrivia/uvadlc-notebooks/tree/main

I think the following is the fundamental for any Neural Network template

```python
import torch.nn as nn
import torch.nn.functional as F

class MyModule(nn.Module):
	def __init__(self):
		super().__init__()
		pass
	def forward(self, x):
		pass
```

 In this code snippet, in the forward function, all the calculations of the module take place. It gets executed when it is called like this -- `nn=MyModule(); nn(x)`

And, in the init function, parameters of the module is defined with `nn.Parameter` or more modules are defined if it needs to be used in the forward function.

The backward calc is done automatically, I don't know what that means yet. Also, I have yet no idea as to why the hidden layer in the midst is necessary and why is there a tanh activation function in each hidden neuron. There are 4 hidden neurons in the hidden layer and two input neurons in the input layer. The blue orbs are input neuron, white orbs are hidden neurons in activation fuctions and red ord is the output neuron.

![[nnXOR.svg]]

This figure can be defined by the following

```python
class SimpleClassifier(nn.Modules):
	def __init__(self, num_inputs, num_hiddens, num_outputs):
		super().__init__()
		self.linear1(num_inputs, num_hiddens)
		self.act_fn(nn.Tanh)
		self.linear2(num_hiddens, num_outputs)
	def forward(self,x):
		x = self.linear1(x)
		x = self.act_fn(x)
		x = self.linear2(x) 
		return x
```

Interesting thing is, in this example, no sigmoid are applied on the output, bcoz the loss function is more efficient and precise to calculate on the original outputs instead of sigmoid output. I don't know about why yet

Lets print out the modules now.

```python
model = SimpleClassifier(num_inputs=2, num_hiddens=4, num_outputs=1
for a,b in model.named_parameters():
	print(f"a={a},b={b}")
```

This prints out weight and bias of each layer in the model
Like this,
```
a=linear1.weight,b=Parameter containing:
tensor([[ 0.6757, -0.4491],
        [-0.3397, -0.6271],
        [ 0.3152,  0.0851],
        [-0.2822,  0.3233]], requires_grad=True)
a=linear1.bias,b=Parameter containing:
tensor([ 0.1370, -0.0494, -0.1857,  0.2974], requires_grad=True)
a=linear2.weight,b=Parameter containing:
tensor([[-0.0168,  0.0308, -0.1449,  0.4847]], requires_grad=True)
a=linear2.bias,b=Parameter containing:
tensor([0.0322], requires_grad=True)

---
```

If we print out b.shape instead of this

```python
model = SimpleClassifier(num_inputs=2, num_hiddens=4, num_outputs=1
for a,b in model.named_parameters():
	print(f"a={a},b={b.shape}")
```

The following gives,

```
a=linear1.weight,b=torch.Size([4, 2])
a=linear1.bias,b=torch.Size([4])
a=linear2.weight,b=torch.Size([1, 4])
a=linear2.bias,b=torch.Size([1])
```

Notice that, Each linear layer has a weigh matrix of the shape `[output, input]` and a bias of shape `[output]`

`10:42 PM ` I dozed off a bit there

The tanh activation function does not have any parameters, not that I understood that much.

The data loading process --
```python
import torch.utils.data as data
```

There are two classes which are useful for playing with data.
`data.Dataset` and `data.DataLoader` 
`data.Dataset` provides a uniform dataset to access the training or testing data
`data.DataLoader` loads data points by stack from dataset into the training process by batches.

To properly define, a `data.Dataset` two functions have to be specified. They are  `__getitem__` and `__len__`. And the infamous `__init__` will be there by default and `super().__init__()` will define it as the parent function of the class. But I am seeing another function there, `generate_continuous_xor` 