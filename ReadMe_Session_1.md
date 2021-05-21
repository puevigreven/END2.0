### What is a neural network neuron?

1. It takes the inputs and multiplies them by their weights,
2. then it sums them up,
3. after that it applies the activation function to the sum.

The neuron’s goal is to adjust the weights based on lots of examples of inputs and outputs. So let’s say we show the neuron a thousand of examples of drawings of cats and drawings of not cats and for each of those examples we show what features are present and how strongly we are sure they are here. Bases on these thousand of images the neuron decides:
which features are important and positive (for example every drawing of cat had a tail in it so the weight must be positive and large),
which features are not important ( for example only a few drawings had 2 eyes, so the weight must be small),
which features are important and negative (for example every drawing containing a horn has been in fact a drawing of a unicorn not a cat so the weight must be large and negative).
The neuron learns the weights based on the inputs and the desired outputs.


### What is the use of the learning rate?
The learning rate is a tuning parameter in an optimization algorithm that determines the step size at each iteration while moving toward a minimum of a loss function. Since it influences to what extent newly acquired information overrides old information, it metaphorically represents the speed at which a machine learning model "learns". In the adaptive control literature, the learning rate is commonly referred to as gain.


### How are weights initialized?
We almost always initialize all the weights in the model to values drawn randomly from a Gaussian or uniform distribution. The choice of Gaussian or uniform distribution does not seem to matter very much, but has not been exhaustively studied. The scale of the initial distribution, however, does have a large effect on both the outcome of the optimization procedure and on the ability of the network to generalize.

### What is "loss" in a neural network?
Loss is nothing but a prediction error of Neural Net. And the method to calculate the loss is called Loss Function.
In simple words, the Loss is used to calculate the gradients. And gradients are used to update the weights of the Neural Net. This is how a Neural Net is trained.


### What is the "chain rule" in gradient flow?
Gradient Flow Calculus is the set of rules used by the Backprop algorithm to compute gradients. Backprop works by first computing the gradients  ∂L/∂yk,i≤k≤K  at the output of the network (which can be computed using an explicit formula), then propagating or flowing these gradients back into the network. 

Node with inputs  (x,y)  and output  z=h(x,y) , where  h  represents the function performed at the node. Lets assume that the gradient  ∂L∂z  is known. Using the Chain Rule of Differentiation, the gradients  ∂L/∂x  and  ∂L/∂y  can be computed as:
∂L∂/x=∂L/∂z*∂z/∂x
∂L/∂y=∂L/∂z*∂z/∂y
