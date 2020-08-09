# Neural Network

This is a machine learning library written in java. Easy to use, load and save datasets.

- Network operations
```java
// Create a new network
NeuralNetwork neuralNetwork = new NeuralNetwork(32, 16, 16, 16, 32);
```
```java
// Load from file
NeuralNetwork.TRAINING_ITERATIONS = NeuralNetwork.load("C:\\network.nn");
```
```java
// Save to file
...
neuralNetwork.save("C:\\network.nn");
...

```

- Full example:
```java
DataSet dataSet = DataSet.load("D:\\images.txt");

NeuralNetwork neuralNetwork = new NeuralNetwork(dataSet.getInputSize(), 10, 10, 10, dataSet.getTargetSize());
NeuralNetwork.TRAINING_ITERATIONS = 10000000;
neuralNetwork.printNetworkInfo();
neuralNetwork.train(dataSet);
neuralNetwork.save("D:\\wow.txt");
```