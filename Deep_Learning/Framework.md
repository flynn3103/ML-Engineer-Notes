## 1. How to write deep learning Framework ? 
-  Step 1: Define Tensor class with automatic gradient computation (autograd)

- Step 2: Upgrading autograd to support multiuse tensors

- Step 3: Adding support for negation
```py
import numpy as np
class Tensor (object):
    def __init__(self, data, 
                creators=None, # creators is a list containing any tensors used in the creation of the current tensor
                creation_op=None, # creation_op is a related feature that stores the instructions creators used in the creation process
                autograd=False,
                id=None):

        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        """
        You create a `self.children` counter that counts the number of gradients received 
        from each child during backpropagation. This way, you also prevent a variable from 
        accidentally backpropagating from the same child twice (which throws an exception)
        """
        if id is None:
            id = np.random.randint(0,10000)
        self.id = id

        if creators is not None:
            for c in creators:
                # Keeps track of how many children a tensor has
                if self.id not in c.children:
                    c.children[self.id] = 1
                else:
                    c.children[self.id] += 1
    """
    Create new function `all_children_grads_accounted_for()` 
    => to compute whether a tensor has received gradients from 
    all of its children 
    """
    def all_children_grads_accounted_for(self):
        for id,cnt in self.children.items():
            """Checks whether a tensor has received the correct 
            number of gradients from each child"""
            if(cnt != 0):
                return False
        return True

    # Adding support for additional functions
    def __add__(self, other):
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                            autograd=True,
                            creators=[self,other],
                            creation_op="add")
        return Tensor(self.data + other.data)

    
    def __mul__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                            autograd=True,
                            creators=[self,other],
                            creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        if(self.autograd):
            return Tensor(self.data.sum(dim),
                            autograd=True,
                            creators=[self],
                            creation_op="sum_"+str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim,copies):
        trans_cmd = list(range(0,len(self.data.shape)))
        trans_cmd.insert(dim,len(self.data.shape))
        new_shape = list(self.data.shape) + [copies]
        new_data = self.data.repeat(copies).reshape(new_shape)
        new_data = new_data.transpose(trans_cmd)
        if(self.autograd):
            return Tensor(new_data,
                            autograd=True,
                            creators=[self],
                            creation_op="expand_"+str(dim))
        return Tensor(new_data)

    def transpose(self):
        if(self.autograd):
            return Tensor(self.data.transpose(),
                            autograd=True,
                            creators=[self],
                            creation_op="transpose")
        return Tensor(self.data.transpose())

    def mm(self, x): #dot matrix
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                            autograd=True,
                            creators=[self,x],
                            creation_op="mm")
        return Tensor(self.data.dot(x.data))
        
    # Adding support for negation
    def __neg__(self):
        if(self.autograd):
            return Tensor(self.data * -1,
                            autograd=True,
                            creators=[self],
                            creation_op="neg")
        return Tensor(self.data * -1)

    def __sub__(self, other):
        if(self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                            autograd=True,
                            creators=[self,other],
                            creation_op="sub")
        return Tensor(self.data - other.data)
    
    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad_origin is not None:
                """
                Checks to make sure you can backpropagate or whether you're waiting for a gradient, 
                in which case decrement the counter
                """
                if self.children[grad_origin.id == 0]:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
            # Accumulates gradients from several children
            if self.grad is None:
                self.grad = grad
            else:
                self.grad += grad
            # Begin actual backpropagation
            if (self.creators is not None and 
               (self.all_children_grads_accounted_for() or
                grad_origin is None)):
                if (self.creation_op == "add"):
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                if(self.creation_op == "sub"):
                    new = Tensor(self.grad.data)
                    self.creators[0].backward(new, self)
                    new = Tensor(self.grad.__neg__().data)
                    self.creators[1].backward(, self)
                if(self.creation_op == "mul"):
                    new = self.grad * self.creators[1]
                    self.creators[0].backward(new , self)
                    new = self.grad * self.creators[0]
                    self.creators[1].backward(new, self)
                if(self.creation_op == "mm"):
                    # Usually an activation 
                    act = self.creators[0]
                    weights = self.creators[1]
                    new = self.grad.mm(weights.transpose())
                    act.backward(new)
                    new = self.grad.transpose().mm(act).transpose()
                    weights.backward(new)
                if(self.creation_op == "transpose"):
                    self.creators[0].backward(self.grad.transpose())
							
                if("sum" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    ds = self.creators[0].data.shape[dim]
                    self.creators[0].backward(self.grad.expand(dim,ds))
                    
                if("expand" in self.creation_op):
                    dim = int(self.creation_op.split("_")[1])
                    self.creators[0].backward(self.grad.sum(dim))

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
```


### Step 5: Using autograd to train a neural network

### Step 6: Adding automatic optimization

### Step 7: Adding support for layer types

### Step 8: Layers that contain layers

### Step 9: Loss-function layers