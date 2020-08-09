package neural;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

import javax.imageio.ImageIO;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

/**
 * Represents a training data or validation data.
 */
public class Data {
	
	private Matrix inputs;
	private Matrix target;
	private String label;
	
	public Data(Matrix inputs, Matrix targets) {
		this.inputs = inputs;
		this.target = targets;
	}
	
	public Data(Matrix inputs) {
		this.inputs = inputs;
	}
	
	public Data(double... inputs) {
		this.inputs = Matrix.createVector(inputs);
	}
	
	public Matrix getInputs() {
		return inputs;
	}

	public void setInputs(Matrix inputs) {
		this.inputs = inputs;
	}

	public Matrix getTarget() {
		return target;
	}

	public void setTarget(Matrix target) {
		this.target = target;
	}
	
	public String getLabel() {
		return label;
	}

	public void setLabel(String label) {
		this.label = label;
	}

	/**
	 * Loads a black and white image.
	 * @param path The location of the file
	 * @return Data object
	 */
	public static Data loadImage(String path) {
		Data d = null;
		try {
			byte[] bytes = Files.readAllBytes(new File(path).toPath());
			ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
			BufferedImage image = ImageIO.read(bais);
			Matrix m = new Matrix(image.getHeight(), image.getWidth());
			for (int i=0; i<image.getHeight(); i++) {
				for (int j=0; j<image.getWidth(); j++) {
					Color mycolor = new Color(image.getRGB(j, i));
					m.setValue(i, j, ((double)mycolor.getRed() + (double)mycolor.getBlue() + (double)mycolor.getGreen()) / 3);
				}
			}
			d = new Data(m);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return d;
	}
	
	/**
	 * Loads a black and white image with label.
	 * @param path The location of the file
	 * @param numOfLabels Number of labels
	 * @param labelIndex Current index
	 * @param label Label text
	 * @return Data object
	 */
	public static Data loadImage(String path, int numOfLabels, int labelIndex, String label) {
		Data d = null;
		try {
			byte[] bytes = Files.readAllBytes(new File(path).toPath());
			ByteArrayInputStream bais = new ByteArrayInputStream(bytes);
			BufferedImage image = ImageIO.read(bais);
			Matrix inputs = new Matrix(image.getHeight(), image.getWidth());
			for (int i=0; i<image.getHeight(); i++) {
				for (int j=0; j<image.getWidth(); j++) {
					Color mycolor = new Color(image.getRGB(j, i));
					inputs.setValue(i, j, ((double)mycolor.getRed() + (double)mycolor.getBlue() + (double)mycolor.getGreen()) / 3);
				}
			}
			Matrix targets = Data.getTargetsFromLabelIndex(numOfLabels, labelIndex);
			d = new Data(inputs, targets);
			d.setLabel(label);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		return d;
	}
	
	/**
	 * Creates a Matrix object with target values
	 * @param numOfLabels Number of labels
	 * @param labelIndex Current index
	 * @return Matrix object
	 */
	public static Matrix getTargetsFromLabelIndex(int numOfLabels, int labelIndex) {
		double[] targets = new double[numOfLabels];
		for (int i=0; i<targets.length; i++) {
			targets[i] = (i == labelIndex) ? 1 : 0;
		}
		return Matrix.createVector(targets);
	}
	
	public static Matrix loadSoundWav(String path) {
		Matrix m = null;
        try {
            ByteArrayOutputStream baout = new ByteArrayOutputStream();
            AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(new File(path));
            AudioSystem.write(audioInputStream, AudioFileFormat.Type.WAVE, baout);
            audioInputStream.close();
            baout.close();
            byte[] data = baout.toByteArray();
            m = new Matrix(data.length, 1);
    		for (int i=0;i<data.length;i++) {
    			int val = data[i] & 0xFF;
    			m.setValue(i, 0, val);
    		}
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return m;
	}

	public Data copy() {
		Data d = new Data();
		d.setInputs(this.inputs == null ? null : this.inputs.copy());
		d.setTarget(this.target == null ? null : this.target.copy());
		d.setLabel(this.label == null ? null : this.label);
		return d;
	}

}
