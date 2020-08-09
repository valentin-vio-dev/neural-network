package neural;

import java.util.Arrays;
import java.util.Random;

import org.opencv.core.Mat;

public class Matrix {
	
	private double[][] data;
	private int n, m;
	
	public Matrix(int n, int m) {
		this.n = n;
		this.m = m;
		this.data = new double[this.n][this.m];
	}
	
	public Matrix(double...vals) {
		this.data = Matrix.createVector(vals).data;
		this.n = vals.length;
		this.m = 1;
	}
	
	/**
	 * Creates a Matrix object from a double array.
	 * @param array Input
	 * @return A matrix object
	 */
	public static Matrix fromArray(double[][] array) {
		if (array == null) return null;
		Matrix matrix = new Matrix(array.length, array[0].length);
		matrix.setData(array);
		return matrix;
	}
	
	/**
	 * Creates a one dimensional Matrix object from a double array.
	 * @param array Input
	 * @return A matrix object
	 */
	public static Matrix createVector(double... array) {
		if (array == null) return null;
		Matrix matrix = new Matrix(array.length, 1);
		for (int i=0; i<matrix.getN(); i++) {
			matrix.setValue(i, 0, array[i]);
		}
		return matrix;
	}
	
	public void init() {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] = 0.0;
			}
		}
	}
	
	/**
	 * Initializes the matrix by a number.
	 * @param num
	 */
	public void init(double num) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] = num;
			}
		}
	}
	
	/**
	 * Randomizes the current matrix data.
	 */
	public void randomize() {
		Random random = new Random();
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] = random.nextDouble() * 2 - 1;
			}
		}
	}
	
	/**
	 * Randomizes the current matrix data betweem two numbers.
	 * @param min The lower limit
	 * @param max The upper limit
	 */
	public void randomize(int min, int max) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] = Utils.random(min, max);
			}
		}
	}
	
	/**
	 * Adds the parameter number to the elements of the matrix.
	 * @param num
	 */
	public void add(double num) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] += num;
			}
		}
	}
	
	/**
	 * Adds the parameter Matrix to the current matrix.
	 * @param m A Matrix object
	 * @throws Exception
	 */
	public void add(Matrix m) throws Exception {
		if (!(m.getN() == this.n && m.getM() == this.m)) {
			throw new Exception("Dimensions do not matches.");
		}
		
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] += m.get(i, j);
			}
		}
	}
	
	/**
	 * Subtracts the parameter number from the elements of the matrix.
	 * @param num
	 */
	public void subtract(double num) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] -= num;
			}
		}
	}
	
	/**
	 * Subtract the matrices from each other
	 * @param m1
	 * @param m2
	 * @return The result Matrix object
	 * @throws Exception
	 */
	public static Matrix subtract(Matrix m1, Matrix m2) throws Exception {
		if (!(m1.getN() == m2.getN() && m1.getM() == m2.getM())) {
			throw new Exception("Dimensions do not matches.");
		}
		
		Matrix res = new Matrix(m1.getN(), 1);
		
		for (int i=0; i<res.getN(); i++) {
			for (int j=0; j<res.getM(); j++) {
				res.setValue(i, j, m1.get(i, j) - m2.get(i, j));
			}
		}
		
		return res;
	}
	
	/**
	 * Multiply the matrix data by a number
	 * @param num
	 */
	public void multiply(double num) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] *= num;
			}
		}
	}
	
	/**
	 * Multiplies the matrix by left by the input matrix one by one.
	 * @param m Matrix
	 * @throws Exception
	 */
	public void multiplyByLeft(Matrix mat) throws Exception {
		if (this.m != mat.getM() || this.n != mat.getN()) {
			throw new Exception("Matrix dimensions must be the same.");
		}
		
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] *= mat.get(i, j);
			}
		}
	}
	
	/**
	 * Multiplies two matrices.
	 * @param m1
	 * @param m2
	 * @return
	 * @throws Exception
	 */
	public static Matrix multiply(Matrix m1, Matrix m2) throws Exception {
		if (m1.getM() != m2.getN()) {
			throw new Exception("Matrix outer dimensions must be equals!");
		}
		
		Matrix mult = new Matrix(m1.getN(), m2.getM());
		
		for (int i=0; i<mult.getN(); i++) {
			for (int j=0; j<mult.getM(); j++) {
				double[] row = Matrix.getRow(m1, i);
				double[] col = Matrix.getCol(m2, j);
				double val = Matrix.addRowAndCol(row, col);
				mult.setValue(i, j, val);
			}
		}
		return mult;
	}
	
	/**
	 * Calculates the squared sum of the input Matrix
	 * @param m Matrix
	 * @return Squared sum
	 */
	public static double getSquaredSum(Matrix m) {
		double sum = 0;
		for (int i=0; i<m.getN(); i++) {
			sum += Math.pow(m.get(i, 0), 2);
		}
		return sum;
	}
	
	/**
	 * Maps the current activation function
	 */
	public void activation() {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				switch (NeuralNetwork.ACTIVATION_FUNCTION) {
				case SIGMOID:
					this.data[i][j] = ActivationFunctions.sigmoid(this.data[i][j]);
					break;
				case TANH:
					this.data[i][j] = ActivationFunctions.tanh(this.data[i][j]);
					break;
				default:
					break;
				}
			}
		}
	}
	
	/**
	 * Maps the current derivate activation' function
	 */
	public void activationDerivate() {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				switch (NeuralNetwork.ACTIVATION_FUNCTION) {
				case SIGMOID:
					this.data[i][j] = ActivationFunctions.dsigmoid(this.data[i][j]);
					break;
				case TANH:
					this.data[i][j] = ActivationFunctions.dtanh(this.data[i][j]);
					break;
				default:
					break;
				}
			}
		}
	}
	
	/**
	 * Transposes the matrix
	 */
	public void transpose() {
		Matrix newMatrix = new Matrix(this.m, this.n);
		for (int i=0; i<newMatrix.getM(); i++) {
			for (int j=0; j<newMatrix.getN(); j++) {
				newMatrix.setValue(j, i, this.data[i][j]);
			}
		}
		this.data = newMatrix.data;
		int tmp = this.n;
		this.n = this.m;
		this.m = tmp;
	}
	
	/**
	 * Transposes the matrix (static)
	 */
	public static Matrix transpose(Matrix m) {
		Matrix newMatrix = new Matrix(m.getM(), m.getN());
		for (int i=0; i<newMatrix.getM(); i++) {
			for (int j=0; j<newMatrix.getN(); j++) {
				newMatrix.setValue(j, i, m.get(i, j));
			}
		}
		return newMatrix;
	}
	
	public void setValue(int n, int m, double val) {
		this.data[n][m] = val;
	}
	
	/**
	 * Prints the matrix
	 */
	public void print() {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				System.out.printf("%.4f\t", this.data[i][j]);
			}
			
			System.out.print("\n");
			
			if (i >= 50) {
				System.out.println("...");
				break;
			}
		}
		System.out.print("\n");
	}
	
	/**
	 * Prints the matrix with precision
	 * @param prec Precision
	 */
	public void print(int prec) {
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				System.out.printf("%." + prec + "f\t", this.data[i][j]);
			}
			System.out.print("\n");
		}
		System.out.print("\n");
	}
	
	/**
	 * Prints the matrix with shading. Only works with values between 0 and 1
	 */
	public void printShade() {
		String shading = " .:-=+*#%@";
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				double shadeValue = this.data[i][j];
				System.out.print(shading.charAt((int) Math.floor(shadeValue * 9)));
			}
			System.out.print("\n");
		}
		System.out.print("\n");
	}
	
	/**
	 * Returns a Matrix row values
	 * @param m Matrix
	 * @param index Row index
	 * @return Row values
	 */
	public static double[] getRow(Matrix m, int index) {
		double[] row = new double[m.getM()];
		for (int i=0; i<row.length; i++) {
			row[i] = m.get(index, i);
		}
		return row;
	}
	
	/**
	 * Returns a Matrix column values
	 * @param m Matrix
	 * @param index Column index
	 * @return Column values
	 */
	public static double[] getCol(Matrix m, int index) {
		double[] col = new double[m.getN()];
		for (int i=0; i<col.length; i++) {
			col[i] = m.get(i, index);
		}
		return col;
	}
	
	/**
	 * Helper function for the matrix multiplication.
	 * @param row Matrix row values
	 * @param col Matrix column values
	 * @return The calculated result
	 */
	public static double addRowAndCol(double[] row, double[] col) {
		double ret = 0;
		for (int i=0; i<row.length; i++) {
			ret += row[i] * col[i];
		}
		return ret;
	}
	
	/**
	 * Copies a Matrix
	 * @param m Input
	 * @return Matrix object
	 */
	public static Matrix copy(Matrix m) {
		Matrix cop = new Matrix(m.getN(), m.getM());
		cop.setData(m.getData());
		return cop;
	}
	
	/**
	 * Copies the Matrix
	 * @return Matrix object
	 */
	public Matrix copy() {
		Matrix cop = new Matrix(this.n, this.m);
		cop.setData(this.data);
		return cop;
	}
	
	/**
	 * Creates a Matrix object from an OpenCV Mat object
	 * @param mat OpenCV Mat object
	 * @return Matrix object
	 */
	public static Matrix fromMat(Mat mat) {
		Matrix m = new Matrix(mat.rows(), mat.cols());
		for (int i=0; i<m.getN(); i++) {
			for (int j=0; j<m.getM(); j++) {
				double val = mat.get(i, j)[0] / 255;
				m.setValue(i, j, val);
			}
		}
		return m;
	}
	
	/**
	 * Creates a vector from a matrix
	 * @param matrix Matrix
	 * @return A vector
	 */
	public static Matrix createVectorFromMatrix(Matrix matrix) {
		Matrix m = new Matrix(matrix.getM() * matrix.getN(), 1);
		int index = 0;
		for (int i=0; i<matrix.getN(); i++) {
			for (int j=0; j<matrix.getM(); j++) {
				m.setValue(index, 0, matrix.get(i, j));
				index++;
			}
		}
		return m;
	}
	
	/**
	 * @return The max value from the current matrix if it is a one dimensional vector
	 * @throws Exception
	 */
	public int getMaxIndexFromVec() throws Exception {
		if (this.m != 1) {
			throw new Exception("Not an one dimensional vector.");
		}
		
		if (this.m == 1 && this.n == 1) {
			return 0;
		}
		
		int index = -1;
		double max = Double.MIN_VALUE;
		for (int i=0; i<this.n; i++) {
			if (this.data[i][0] > max) {
				max = this.data[i][0];
				index = i;
			}
		}
		return index;
	}
	
	/**
	 * Decreases a matrix by some value
	 * @param m Matrix
	 * @param decrease A number (e.g.: The 2 value means the half of the matrix)
	 * @return A Matrix object
	 */
	public static Matrix decrease(Matrix m, int decrease) {
		Matrix mat = new Matrix(m.getN()/decrease, m.getM()/decrease);
		for (int i=0; i<mat.getN(); i++) {
			for (int j=0; j<mat.getM(); j++) {
				mat.setValue(i, j, m.get(i*decrease, j*decrease));
			}
		}
		return mat;
	}
	
	/**
	 * Decreases a vector by some value
	 * @param m Matrix
	 * @param decrease A number (e.g.: The 2 value means the half of the matrix)
	 * @return A Matrix object
	 * @throws Exception
	 */
	public static Matrix decreaseVector(Matrix m, int decrease) throws Exception {
		if (m.getM() != 1) {
			throw new Exception("The input matrix must be one dimensional");
		}
		
		Matrix mat = new Matrix(m.getN()/decrease, 1);
		for (int i=0; i<mat.getN(); i++) {
			mat.setValue(i, 0, m.get(i*decrease, 0));
		}
		return mat;
	}
	
	/**
	 * Creates a Matrix object from a double list
	 * @param n Number of rows
	 * @param m Number of columns
	 * @param list
	 * @return A Matrix object
	 */
	public static Matrix fromList(int n, int m, double[] list) {
		Matrix mat = new Matrix(n, m);
		int index = 0;
		for (int i=0; i<mat.getN(); i++) {
			for (int j=0; j<mat.getM(); j++) {
				mat.setValue(i, j, list[index]);
				index++;
			}
		}
		return mat;
	}
	
	public static Matrix createMatrixFromVector(Matrix vec, int n, int m) throws Exception {
		if (!vec.isVector()) {
			throw new Exception("Matrix must be one dimensional vector");
		}
		
		return Matrix.fromList(n, m, Matrix.getCol(vec, 0));
	}
	
	/**
	 * @return The maximum value from the matrix.
	 */
	public double getMaxValue() {
		double max = Double.MIN_VALUE;
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				if (this.data[i][j] > max) {
					max = this.data[i][j];
				}
			}
		}
		return max;
	}
	
	/**
	 * Optimize the matrix. All input values will be between 0 and 1.
	 */
	public void optimize() {
		double max = this.getMaxValue();
		for (int i=0; i<this.n; i++) {
			for (int j=0; j<this.m; j++) {
				this.data[i][j] /= max;
			}
		}
	}
		
	public double[][] getData() {
		return this.data;
	}
	
	public double get(int i, int j) {
		return this.data[i][j];
	}

	public void setData(double[][] data) {
		this.data = data;
	}

	public int getN() {
		return n;
	}

	public void setN(int n) {
		this.n = n;
	}

	public int getM() {
		return m;
	}

	public void setM(int m) {
		this.m = m;
	}
	
	public int[] size() {
		return new int[] {this.n, this.m};
	}
	
	/**
	 * Prints the size of the matrix
	 */
	public void printSize() {
		System.out.println(Arrays.toString(this.size()));
	}
	
	/**
	 * @return Number of elements in the matrix
	 */
	public int getElementsCount() {
		return this.n * this.m;
	}
	
	/**
	 * Convert the matrix into a single line text.
	 * @return Converted text
	 */
	public String toLine() {
		Matrix m = Matrix.transpose(Matrix.createVectorFromMatrix(this));
		double[] arr = Matrix.getRow(m, 0);
		return Arrays.toString(arr).replace("[", "").replace("]", "").replaceAll(" ", "");
	}
	
	/**
	 * @return True, if it is a one dimensional vector
	 */
	public boolean isVector() {
		return (this.m == 1) ? true : false;
	}

}
