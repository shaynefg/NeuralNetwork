import numpy as np
from  activation import activation
from losses import lossFunction

class weights(object):
    def __init__(self, input_size, output_size):
      self.weights = np.random.randn(input_size, output_size)*0.01
    
    def get_weights(self,):
      return self.weights 

class biases(object):
    def __init__(self, input_size):
      self.biases = np.transpose(np.random.randn(input_size, 1) - 0.5)
     
    def get_biases(self,):
      return np.array(self.biases)   

class layer(object):
    def __init__(self, layer_size):
        self.layer_size = layer_size

class Network(object):
    def __init__(self, input_size, sizes):
        self.learningRate = 0.1
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.layer_inputs = []
        self.layer_net = []
        self.layer_out = []
        self.layer_in = []

        self.input_size = input_size
        self.sizes = np.concatenate((np.array([self.input_size]), np.array(self.sizes)))
        #print(self.sizes)
        #self.biases = [biases(y) for y in sizes[1:]]
        self.weights = [weights(x,y) for x,y in zip(self.sizes[:-1], self.sizes[1:])]
        #print(len(self.weights))

    def forward(self, input):
      
      i = 0
      
      #self.layer_out.append(input)
      
      out = input
      for i, layer in enumerate(self.weights):
        self.layer_in.append(out)

        if self.sizes[i+1] > 1:
          #print("Layer IN :", out)
          #print("Layer WEIGHTS :", layer.weights)
          net = np.matmul(layer.weights.T, out.T) # + np.transpose(self.biases[i].biases)
        else:
          #print("Layer IN :", out)
          #print("Layer WEIGHTS :", layer.weights)
          net = np.array(np.sum(np.multiply(layer.weights, out)))

        ##print("Layer net",net)


        #print("LAYER NET :", net)
        
        self.layer_net.append(net)
        out = 1 / (1 + np.exp(-net)).T
        self.layer_out.append(out)
        #print("LAYER OUT :", out)

        ##print("L out", out)

        i = i + 1

      #(input)
      ##print("Layer out size :", len(self.layer_out))
      return out

    def backward(self, output_true, input):
      self.input = np.array([input]) 

      #1.Forward pass

      self.layer_net = []
      self.layer_out = []

      self.output = self.forward(self.input)
      #print("Layer out:", self.layer_out[-1])
      #print("Layer Net:", self.layer_net[-1])
      #Last layer 
      outputError = lossFunction().errorPartial(output_true, self.output)
      delta =  np.multiply(outputError.T, activation().sigmoidDerivative(self.layer_net[-1]))
      #print("Ouptut error :", outputError)
      #print("Delta :" ,delta)
      weightError =  self.learningRate * np.multiply(delta, self.layer_in[-1])
      #print("weightError :" ,weightError)
      #print("Weights :", self.weights[-1].weights)
      self.weights[-1].weights =  np.add(self.weights[-1].weights, weightError.T)




      #print("New weights ", self.weights[-1].get_weights())
      for i in reversed(range(0, len(self.weights)-1)):
        #print("Layer : ", i)
        self.layer = self.weights[i]
        #2.Compute error
        #print(self.layer.weights.shape, delta.shape)
        #print("Layer out:", self.layer_out[i])
        #print("Layer Net:", self.layer_net[i])

        #print("SIZE NEXT LAYER :", self.sizes[i+2])

        if delta.shape[0] > self.layer.weights.shape[1]:
           delta = delta[self.layer.weights.shape[1], 1]
        else:
           t = np.zeros((self.layer.weights.shape[1], 1))
           t[0:delta.shape[0],:] = delta
           delta = t

        #print("Weights ,", self.layer.weights.T.shape)
        #print("delta ,", delta.T.shape)

        

        if self.sizes[i+2] > 1:
          delta 
          delta = np.matmul(self.layer.weights, delta) # + np.transpose(self.biases[i].biases)
        else:
          delta = np.array(np.sum(np.multiply(self.layer.weights, delta)))


        #delta =  np.matmul(self.layer.weights, delta.T) * activation().sigmoidDerivative(self.layer_net[i])
        #print(delta.shape)
        #print("Delta", delta)
        #3.Backwards Pass
        weightError =  self.learningRate * np.multiply(delta, self.layer_in[i+1])
        #print("Weight Delta", self.layer_in[i+1])
        
        #4.Compute Gradient
        #weightError = np.matmul(np.transpose(self.layer_inputs[i]), d_outputError)
        #5.Update, Weights
        self.weights[i].weights +=  weightError

        ##print(self.weights[i].weights)
        #output_true = weightError
        ##print(i)
        #self.output =  self.layer_out[i]
        # outputError = lossFunction().errorPartial(inputError, self.output) 

      


if __name__ == "__main__":
  net = Network(2, [30,20, 4])
  # net.weights[0].weights = np.array([[0.14, .15],
  #                                    [.27, .14]])

  # net.weights[1].weights = np.array([[0.23, .21]]).T


  # #print(len(net.weights))

  # #print(net.forward(np.array([[2, 3]])) )
  # #print(net.backward([1], [2,3]))

  # #print(net.weights[0].weights)
  # #print(net.weights[1].weights)

  for i in range(0, 100):

    
    print("Iteration ", i)
    net.backward([1,0,0,0],[0, 0])
    print(net.forward(np.array([[0,0]])) )

    net.backward([0,1,0,0],[0, 1])
    print(net.forward(np.array([[0,1]])) )

    net.backward([0,0,1,0],[1, 0])
    print(net.forward(np.array([[1,0]])) )

    net.backward([0,0,0,1], [1,1])
    print(net.forward(np.array([[1,1]])) )
