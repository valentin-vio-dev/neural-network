package neural;

/**
 * Activation function collection for the Neural Network.
 */
public class ActivationFunctions {
	
	public static enum Function {
		SIGMOID,
		TANH
	};
	
	public static double sigmoid(double x) {
		return 1 / (1 + Math.pow(Math.E, -x));
	}
	
	public static double dsigmoid(double x) {
		return x * (1 - x);
	}
	
	public static double tanh(double x) {
		return (Math.pow(Math.E, x) - Math.pow(Math.E, -x))/(Math.pow(Math.E, x) + Math.pow(Math.E, -x));
	}
	
	public static double dtanh(double x) {
		return 1 - ActivationFunctions.tanh(ActivationFunctions.tanh(x));
	}
}
