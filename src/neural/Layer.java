package neural;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import neural.NeuralNetwork.Colors;

/**
 * Represents a network layer in the Neural Network.
 */
public class Layer {
	
	private Matrix data;
	private Layer layerPrev;
	private Layer layerNext;
	private Matrix bias;
	private Matrix calculatedFeed;
	private int index;
	
	public Layer(Matrix data) {
		this.data = data;
		this.bias = new Matrix(data.getN(), 1);
		this.bias.randomize();
	}
	
	public Matrix getBias() {
		return bias;
	}

	public void setBias(Matrix bias) {
		this.bias = bias;
	}

	public Matrix getData() {
		return this.data;
	}

	public void setData(Matrix data) {
		this.data = data;
	}

	public Layer getLayerPrev() {
		return layerPrev;
	}

	public void setLayerPrev(Layer layerPrev) {
		this.layerPrev = layerPrev;
	}

	public Layer getLayerNext() {
		return layerNext;
	}

	public void setLayerNext(Layer layerNext) {
		this.layerNext = layerNext;
	}

	public void randomize() {
		this.data.randomize();
	}
	
	public void print() {
		this.data.print();
	}

	public int getIndex() {
		return this.index;
	}

	public void setIndex(int index) {
		this.index = index;
	}
	
	public Matrix getCalculatedFeed() {
		return calculatedFeed;
	}

	public void setCalculatedFeed(Matrix calculatedFeed) {
		this.calculatedFeed = calculatedFeed;
	}

	/**
	 * Feed forwards the inputs and calculates the prediction by matrix operations.
	 * @param start True, if it is in starting stage
	 * @param trainingData The training data
	 * @param resultMatrix A matrix for side calculations
	 * @return A Matrix with the predicted data
	 * @throws Exception
	 */
	public Matrix feedForward(boolean start, Data trainingData, Matrix resultMatrix) throws Exception {	
		Matrix nextInput = start ? trainingData.getInputs() : resultMatrix;
		Matrix multiplied = Matrix.multiply(this.data, nextInput);
		multiplied.add(this.bias);
		multiplied.activation();
		this.calculatedFeed = multiplied;
		
		if (this.layerNext == null) return multiplied;
		return this.layerNext.feedForward(false, trainingData, multiplied);
	}
	
	/**
	 * Back propagates the outputs and calculates the new network weights and biases.
	 * @param start True, if it is in starting stage
	 * @param output The output of the feedForward algorithm
	 * @param loss The loss matrix
	 * @param trainingData The training data
	 * @return True, if back propagate ended.
	 * @throws Exception
	 */
	public boolean backPropagate(boolean start, Matrix output, Matrix loss, Data trainingData) throws Exception {
		Matrix gradient = Matrix.copy(start ? output : this.calculatedFeed);
		gradient.activationDerivate();
		gradient.multiplyByLeft(loss);
		gradient.multiply(NeuralNetwork.LEARNING_RATE);
	
		Matrix transposedLayer = Matrix.copy(this.layerPrev == null ? trainingData.getInputs() : this.layerPrev.getCalculatedFeed());
		transposedLayer.transpose();
		Matrix delta = Matrix.multiply(gradient, transposedLayer);
		this.data.add(delta);
		this.bias.add(gradient);
		
		Matrix prevT = Matrix.copy(this.data);
		prevT.transpose();
		Matrix nextLoss = Matrix.multiply(prevT, loss);
		
		if (this.layerPrev == null) return true;
		return this.layerPrev.backPropagate(false, null, nextLoss, trainingData);
	}
	
	/**
	 * Convert the weight matrix into a single line text.
	 * @return Converted text
	 */
	public String weightsToLine() {
		return this.data.toLine();
	}
	
	/**
	 * Convert the bias matrix into a single line text.
	 * @return Converted text
	 */
	public String biasToLine() {
		return this.bias.toLine();
	}
	
	public void savePng(String path) {
	    BufferedImage image = new BufferedImage(this.data.getM(), this.data.getN(), BufferedImage.TYPE_INT_RGB);
	    for (int i=0; i<image.getHeight(); i++) {
	    	for (int j=0; j<image.getWidth(); j++) {
	    		int c = Math.min(255, (int)Math.floor(Math.abs(this.data.get(i, j)) * 255));
	    		int r = NeuralNetwork.IMAGE_COLOR == Colors.RED ? c : 0;
	    		int g = NeuralNetwork.IMAGE_COLOR == Colors.GREEN ? c : 0;
	    		int b = NeuralNetwork.IMAGE_COLOR == Colors.BLUE ? c : 0;
	    		image.setRGB(j, i, new Color(r, g, b).getRGB());
	    	}
	    }
	   

	    File ImageFile = new File(path);
	    try {
	        ImageIO.write(image, "png", ImageFile);
	    } catch (IOException e) {
	        e.printStackTrace();
	    }
	}
	
	public BufferedImage convertToImage() {
		BufferedImage image = new BufferedImage(this.data.getM(), this.data.getN(), BufferedImage.TYPE_INT_RGB);
	    for (int i=0; i<image.getHeight(); i++) {
	    	for (int j=0; j<image.getWidth(); j++) {
	    		image.setRGB(j, i, (int)Math.floor(this.data.get(i, j) * 255));
	    	}
	    }
	    return image;
	}

}
