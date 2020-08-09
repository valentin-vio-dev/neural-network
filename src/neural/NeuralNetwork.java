package neural;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Random;
import java.util.Scanner;

import javax.imageio.ImageIO;

import neural.ActivationFunctions.Function;

/**
 * Represents a NeuralNetwork model.
 */
public class NeuralNetwork {
	
	public static double LEARNING_RATE = 0.03;
	public static int TRAINING_ITERATIONS = 100;
	public static boolean INFO_ENABLED = true;
	public static ActivationFunctions.Function ACTIVATION_FUNCTION = ActivationFunctions.Function.SIGMOID;
	public static Colors IMAGE_COLOR = Colors.GREEN;
	
	private int[] layerSizes;
	private Layer[] layers;
	
	public NeuralNetwork(int... layerSizes) { 
		this.layerSizes = layerSizes;
		this.constructLayers();
		this.setNeigborLayers();
	}
	
	public NeuralNetwork() { }
	
	public int[] getLayerSizes() {
		return layerSizes;
	}

	public void setLayerSizes(int[] layerSizes) {
		this.layerSizes = layerSizes;
	}

	/**
	 * Construct all hidden layers from the layer sizes.
	 */
	private void constructLayers() {
		this.layers = new Layer[this.layerSizes.length-1];
		int index = 0;
		for (int i=0; i<layers.length; i++) {
			Matrix layerMatrix = new Matrix(this.layerSizes[i+1], this.layerSizes[i]);
			this.layers[i] = new Layer(layerMatrix);
			this.layers[i].setIndex(index);
			this.layers[i].randomize();
			index++;
		}
	}
	
	/**
	 * Iterates through all layers and sets neighbor references.
	 */
	private void setNeigborLayers() {
		for (int i=0; i<layers.length; i++) {
			if (i == 0) {
				this.layers[i].setLayerNext(this.layers[i+1]);
				this.layers[i].setLayerPrev(null);
			} else if (i == layers.length - 1) {
				this.layers[i].setLayerNext(null);
				this.layers[i].setLayerPrev(this.layers[layers.length - 2]);
			} else {
				this.layers[i].setLayerNext(this.layers[i+1]);
				this.layers[i].setLayerPrev(this.layers[i-1]);
			}
		}
	}
	
	/**
	 * Trains the network with a Data.
	 * @param data Training data
	 * @return A Matrix with the calculated errors.
	 */
	public Matrix train(Data data) {
		Matrix result = null;
		try {
			if (!data.getInputs().isVector()) {
				data.setInputs(Matrix.createVectorFromMatrix(data.getInputs()));
			}
			
			if (!data.getTarget().isVector()) {
				data.setTarget(Matrix.createVectorFromMatrix(data.getTarget()));
			}
			
			Matrix output = layers[0].feedForward(true, data, null);
			Matrix loss = Matrix.subtract(data.getTarget(), output);
			result = loss.copy();
			this.getLastLayer().backPropagate(true, output, loss, data);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	/**
	 * Trains the network with a DataSet.
	 * @param dataSet Training dataset
	 */
	public void train(DataSet dataSet) {
		Random random = new Random();
		printInfo("Training start.");
		for (int i=0; i<NeuralNetwork.TRAINING_ITERATIONS; i++) {
			double percent = ((double)i / NeuralNetwork.TRAINING_ITERATIONS * 100);
			int r = random.nextInt(dataSet.size());
			Data d = dataSet.getData(r).copy();
			if (!d.getInputs().isVector()) {
				d.setInputs(Matrix.createVectorFromMatrix(d.getInputs()));
			}
			Matrix trainLoss = this.train(d);
			
			if (percent % 1 == 0) {
				printInfo(percent + " %");
				printLoss(Matrix.getSquaredSum(trainLoss));
			}
		}
		printInfo("Training complete.");
	}
	
	/**
	 * Tests the network with a DataSet.
	 * @param dataSet Test dataset
	 * @return A value which represents the network's accuracy.
	 */
	public double test(DataSet dataSet) {
		printInfo("Test start.");
		int correctPredict = 0;
		for (int i=0; i<dataSet.size(); i++) {
			Data d = dataSet.getData(i);
			if (!d.getInputs().isVector()) {
				d.setInputs(Matrix.createVectorFromMatrix(d.getInputs()));
			}
			try {
				double target = dataSet.getData(i).getTarget().getMaxIndexFromVec();
				Matrix predictMatrix = predict(d);
				double predicted = predictMatrix.getMaxIndexFromVec();
				double maxVal = predictMatrix.getMaxValue();
				if (target == predicted) {
					correctPredict++;
				}
				String label = dataSet.getData(i).getLabel() == null ? "-" : dataSet.getData(i).getLabel();
				printInfo("Label: " + label + "\t Target: " + target + "\t" + " Predict: " + predicted + " (" + maxVal + ")");
			} catch (Exception e) {
				e.printStackTrace();
			}	
		}
		double acc = (double)correctPredict / (double)dataSet.size() * 100;
		printInfo("Accuracy: " + acc + " %");
		printInfo("Test complete.");
		return acc;
	}
	
	/**
	 * Predicts a data.
	 * @param data Test data
	 * @return A Matrix with the calculated predictions.
	 */
	public Matrix predict(Data data) {
		Matrix result = null;
		try {
			if (!data.getInputs().isVector()) {
				data.setInputs(Matrix.createVectorFromMatrix(data.getInputs()));
			}
			result = layers[0].feedForward(true, data, null);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}
	
	/**
	 * @return The last layer of the network.
	 */
	private Layer getLastLayer() {
		return this.layers[this.layers.length - 1];
	}
	
	/**
	 * Prints the shape of the network layers.
	 */
	public void printShape() {		
		for (int i=0; i<layers.length; i++) {
			System.out.println("[" + (layers[i].getIndex()) + ". layer]");
			this.layers[i].print();	
		}
	}
	
	/**
	 * Prints some info about the network.
	 */
	public void printNetworkInfo() {
		printInfo("Network info");
		System.out.println("Layer sizes:\t\t" + Arrays.toString(this.layerSizes) + "(" + this.layerSizes.length + ")");
		System.out.println("Input size:\t\t" + this.getInputsSize());
		System.out.println("Target size:\t\t" + this.getTargetsSize());
		System.out.println("Learning rate:\t\t" + NeuralNetwork.LEARNING_RATE);
		System.out.println("Training iterations:\t" + String.format("%,d", NeuralNetwork.TRAINING_ITERATIONS));
		System.out.println("Activation function:\t" + NeuralNetwork.ACTIVATION_FUNCTION);
		System.out.println("Layers:");
		printShape();
		System.out.println("---\n");
	}
	
	/**
	 * Saves the network's weights and biases to the selected location.
	 * @param path The location of the new file
	 */
	public void save(String path) {
		try {
			FileWriter myWriter = new FileWriter(path);
			myWriter.write(keyValue("layer_sizes", Arrays.toString(this.layerSizes).replace("[", "").replace("]", "").replaceAll(" ", "")));
			myWriter.write(keyValue("activation_function", NeuralNetwork.ACTIVATION_FUNCTION.name()));
			for (int i=0; i<layers.length; i++) {
				myWriter.write(keyValue("layer_w::" + i, layers[i].weightsToLine()));
				myWriter.write(keyValue("layer_b::" + i, layers[i].biasToLine()));
			}
			myWriter.close();
			printInfo("Network successfully saved. [" + path + "]");
		} catch (IOException e) {
			printInfo("An error occurred while saving.");
	    }
	}
	
	/**
	 * Loads the network's weights and biases from the selected location.
	 * @param path The location of the file
	 * @return The loaded NeuralNetwork object.
	 */
	public static NeuralNetwork load(String path) {
		NeuralNetwork network = new NeuralNetwork();
		try {
			Scanner scanner = new Scanner(new File(path));
			while (scanner.hasNextLine()) {
		        String data = scanner.nextLine();
		        String key = data.split("=")[0];
		        String val = data.split("=")[1];
		        
		        if (key.equals("LAYER_SIZES")) {
		        	int[] layers = Arrays.asList(val.split(",")).stream().mapToInt(Integer::parseInt).toArray();
		        	network.setLayerSizes(layers);
		        	network.constructLayers();
		        	network.setNeigborLayers();
		        } else if (key.equals("ACTIVATION_FUNCTION")) {
		        	NeuralNetwork.ACTIVATION_FUNCTION = Function.valueOf(val);
		        } else if (key.startsWith("LAYER_W")) {
		        	int indexOfLayer = Integer.parseInt(key.split("::")[1]);
		        	int n = network.layers[indexOfLayer].getData().getN();
		        	int m = network.layers[indexOfLayer].getData().getM();
		        	double[] values = Arrays.asList(val.split(",")).stream().mapToDouble(Double::parseDouble).toArray();
		        	network.layers[indexOfLayer].setData(Matrix.fromList(n, m, values));
		        } else if (key.startsWith("LAYER_B")) {
		        	int indexOfLayer = Integer.parseInt(key.split("::")[1]);
		        	int n = network.layers[indexOfLayer].getBias().getN();
		        	int m = network.layers[indexOfLayer].getBias().getM();
		        	double[] values = Arrays.asList(val.split(",")).stream().mapToDouble(Double::parseDouble).toArray();
		        	network.layers[indexOfLayer].setBias(Matrix.fromList(n, m, values));
		        }
		      }
			scanner.close();
			staticPrintInfo("Network successfully loaded. [" + path + "]");
		} catch (Exception e) {
			staticPrintInfo("An error occurred while loading.");
			return null;
		}
		
		return network;
	}
	
	/**
	 * Concatenates a key and value pair to a String.
	 * @param key String key
	 * @param value String value
	 * @return Concatenated String
	 */
	private String keyValue(String key, String value) {
		return key.toUpperCase() + "=" + value + "\n";
	}
	
	/**
	 * Prints a text to the console.
	 * @param text A text
	 */
	private void printInfo(String text) {
		if (!NeuralNetwork.INFO_ENABLED) return;
		String time = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		System.out.println("[NeuralNetwork | " + time + "] " + text);
	}
	
	/**
	 * Prints a text to the console (static).
	 * @param text A text
	 */
	private static void staticPrintInfo(String text) {
		if (!NeuralNetwork.INFO_ENABLED) return;
		String time = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		System.out.println("[NeuralNetwork | " + time + "] " + text);
	}
	
	/**
	 * Prints loss to the console.
	 * @param val Loss
	 */
	private void printLoss(double val) {
		if (!NeuralNetwork.INFO_ENABLED) return;
		String time = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		System.out.printf("[NeuralNetwork | %s] Loss: %.16f\n", time, val);
	}

	public Layer getLayer(int index) {
		return this.layers[index];
	}
	
	public void saveLayersPng(String path) {
		int width = Integer.MIN_VALUE;
		int height = 0;
		for (Layer l: this.layers) {
			if (l.getData().getN() > width) {
				width = l.getData().getM();
			}
			height += l.getData().getN();
		}
		
		BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		
		int y = 0;
		for(Layer l: layers) {
			for (int i=0; i<l.getData().getN(); i++) {
				if (i == 0) {
					for (int j=0; j<image.getWidth(); j++) {
			    		int c = 10;
			    		image.setRGB(j, y+i, new Color(c, c, c).getRGB());
			    	}
					continue;
				} 
		    	for (int j=0; j<l.getData().getM(); j++) {
		    		int c = Math.min(255, (int)Math.floor(Math.abs(l.getData().get(i, j)) * 255));
		    		int r = IMAGE_COLOR == Colors.RED ? c : 0;
		    		int g = IMAGE_COLOR == Colors.GREEN ? c : 0;
		    		int b = IMAGE_COLOR == Colors.BLUE ? c : 0;
		    		image.setRGB(j, y+i, new Color(r, g, b).getRGB());
		    	}
		    }
			y += l.getData().getN();
		}
		
		File ImageFile = new File(path);
	    try {
	        ImageIO.write(image, "png", ImageFile);
	    } catch (IOException e) {
	    	printInfo("An error occurred while saving.");
	    }
	    
	    printInfo("Network successfully saved as image. [" + path + "]");
	}
	
	public void predictToPng(Data data, String path) {
		Matrix predict = this.predict(data.copy());
		try {
			predict = Matrix.createMatrixFromVector(predict.copy(), data.getInputs().getN(), data.getInputs().getM());
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		BufferedImage image = new BufferedImage(predict.getM(), predict.getN(), BufferedImage.TYPE_INT_RGB);
		
		for (int i=0; i<image.getHeight(); i++) {
	    	for (int j=0; j<image.getWidth(); j++) {
	    		int c = Math.min(255, (int)Math.floor(Math.abs(predict.get(i, j)) * 255));
	    		int r = IMAGE_COLOR == Colors.RED ? c : 0;
	    		int g = IMAGE_COLOR == Colors.GREEN ? c : 0;
	    		int b = IMAGE_COLOR == Colors.BLUE ? c : 0;
	    		image.setRGB(j, i, new Color(r, g, b).getRGB());
	    	}
	    }
		
		image = Utils.scale(image, image.getWidth() * 8, image.getHeight() * 8);
		
		File ImageFile = new File(path);
	    try {
	        ImageIO.write(image, "png", ImageFile);
	    } catch (IOException e) {
	    	printInfo("An error occurred while saving.");
	    }
	    
	    printInfo("Predict successfully saved as image. [" + path + "]");
	}
	
	public int getInputsSize() {
		return this.layerSizes[0];
	}
	
	public int getTargetsSize() {
		return this.layerSizes[this.layerSizes.length - 1];
	}
	
	public enum Colors {
		RED, GREEN, BLUE
	}
}
