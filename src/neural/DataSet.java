package neural;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Scanner;

public class DataSet {
	
	private List<Data> data;
	private List<String> labels;
	
	public DataSet() {
		this.data = new ArrayList<>();
		this.labels = new ArrayList<>();
	}
	
	/**
	 * Optimize the dataset. All input values will be between 0 and 1.
	 */
	public void optimize() {
		for(Data d: this.data) {
			d.getInputs().optimize();
		}
	}
	
	/**
	 * Loads a dataset from a specified location.
	 * @param path Location of the dataset
	 * @return Dataset object
	 */
	public static DataSet loadImages(String path) {
		DataSet dataSet = new DataSet();
		try {
			if (path.equals("")) {
				throw new Exception();
			}
			
			int labelIndex = 0;
			File dir = new File(path);
			
			for(String dirName: dir.list()) {
				String label = dirName;
				dataSet.addLabel(label);
				String subDirPath = dir.getAbsolutePath() + "\\" + dirName;
				File subDir = new File(subDirPath);
				
				if (subDir.isFile()) {
					staticPrintInfo("Load " + subDirPath);
					dataSet.addData(Data.loadImage(subDirPath, dir.list().length, labelIndex, label));
				} else {
					for(String dataFileName: subDir.list()) {
						String file = subDirPath + "\\" + dataFileName;
						staticPrintInfo("Load " + file);
						dataSet.addData(Data.loadImage(file, dir.list().length, labelIndex, label));
					}
					labelIndex++;
				}
				
				
				
			}
			staticPrintInfo("Loaded complete. Loaded " + dataSet.size() + " files.");
		} catch (Exception e) {
			staticPrintInfo("An error occurred while loading.");
		}
		
		return dataSet;
	}
	
	public static DataSet load(String path) {
		DataSet dataSet = new DataSet();
		staticPrintInfo("Dataset loading...");
		try {
			Scanner scanner = new Scanner(new File(path));
			while (scanner.hasNextLine()) {
		        String inputs = scanner.nextLine();
		        String targets = scanner.nextLine();
		        double[] inputValues = Arrays.asList(inputs.split(",")).stream().mapToDouble(Double::parseDouble).toArray();
		        double[] targetValues = Arrays.asList(targets.split(",")).stream().mapToDouble(Double::parseDouble).toArray();
		        Matrix inputMatrix = Matrix.createVector(inputValues);
		        Matrix targetMatrix = Matrix.createVector(targetValues);
		        dataSet.addData(new Data(inputMatrix, targetMatrix));
			}
			scanner.close();
			staticPrintInfo("Dataset successfully loaded. [" + path + "]");
		} catch (Exception e) {
			staticPrintInfo("An error occurred while loading.");
			return null;
		}
		return dataSet;
	}
	
	public void save(String path) {
		printInfo("Dataset saving...");
		try {
			FileWriter myWriter = new FileWriter(path);
			for(Data d: this.data) {
				myWriter.write(d.getInputs().toLine()+"\n");
				myWriter.write(d.getTarget().toLine()+"\n");
			}
			myWriter.close();
			printInfo("Dataset successfully saved. [" + path + "]");
		} catch (IOException e) {
			printInfo("An error occurred while saving.");
	    }
	}
	
	/**
	 * Decreases an image size.
	 * @param val The amount of decrease
	 */
	public void decreaseImage(int val) {
		for(Data d: this.data) {
			d.setInputs(Matrix.decrease(d.getInputs(), val));
		}
	}
	
	/**
	 * Decreases a vector data.
	 * @param val The amount of decrease
	 */
	public void decreaseData(int val) {
		for(Data d: this.data) {
			try {
				d.setInputs(Matrix.decreaseVector(d.getInputs(), val));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
	
	/**
	 * Decreases an image size or a vector data.
	 * @param val The amount of decrease
	 */
	public void decrease(int val) {
		if (this.data.get(0).getInputs().isVector()) {
			this.decreaseData(val);
		} else {
			this.decreaseImage(val);
		}
	}
	
	public List<Data> getData() {
		return data;
	}

	public List<String> getLabels() {
		return labels;
	}

	public void setLabels(List<String> labels) {
		this.labels = labels;
	}

	public void setData(List<Data> data) {
		this.data = data;
	}

	public Data getData(int index) {
		return this.data.get(index);
	}
	
	public int addData(Data data) {
		this.data.add(data);
		return this.data.size();
	}
	
	public int addLabel(String label) {
		this.labels.add(label);
		return this.labels.size();
	}
	
	public int size() {
		return this.data.size();
	}
	
	public int getInputSize() {
		return this.data.get(0).getInputs().getElementsCount();
	}
	
	public int getTargetSize() {
		return this.data.get(0).getTarget().getElementsCount();
	}
	
	/**
	 * Prints a text to the console.
	 * @param text A text
	 */
	private void printInfo(String text) {
		if (!NeuralNetwork.INFO_ENABLED) return;
		String time = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		System.out.println("[DataSet | " + time + "] " + text);
	}
	
	/**
	 * Prints a text to the console.
	 * @param text A text
	 */
	private static void staticPrintInfo(String text) {
		if (!NeuralNetwork.INFO_ENABLED) return;
		String time = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime());
		System.out.println("[DataSet | " + time + "] " + text);
	}
	
	public void printData() {
		for(Data d: this.data) {
			d.getInputs().print();
			d.getTarget().print();
		}
	}
	
	public void setInputsAsTarget() {
		for (Data d: this.data) {
			d.setTarget(d.getInputs());
		}
	}

}
