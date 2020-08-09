Neural Network

```javas
DataSet dataSet = DataSet.load("D:\\images.txt");

NeuralNetwork neuralNetwork = new NeuralNetwork(dataSet.getInputSize(), 10, 10, 10, dataSet.getTargetSize());
NeuralNetwork.TRAINING_ITERATIONS = 10000000;
neuralNetwork.printNetworkInfo();
neuralNetwork.train(dataSet);
neuralNetwork.save("D:\\wow.txt");
```