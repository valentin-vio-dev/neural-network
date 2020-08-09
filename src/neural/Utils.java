package neural;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Random;

public class Utils {
	
	/**
	 * @param min
	 * @param max
	 * @return A random number between min and max
	 */
	public static double random(int min, int max) {
		Random random = new Random();
		return random.nextInt(max - min) + min;
	}

	public static BufferedImage scale(BufferedImage imageToScale, int dWidth, int dHeight) {
        BufferedImage scaledImage = null;
        if (imageToScale != null) {
            scaledImage = new BufferedImage(dWidth, dHeight, imageToScale.getType());
            Graphics2D graphics2D = scaledImage.createGraphics();
            graphics2D.drawImage(imageToScale, 0, 0, dWidth, dHeight, null);
            graphics2D.dispose();
        }
        return scaledImage;
    }
	
	public static BufferedImage scale(BufferedImage imageToScale, int scale) {
        BufferedImage scaledImage = null;
        if (imageToScale != null) {
            scaledImage = new BufferedImage(scale, scale, imageToScale.getType());
            Graphics2D graphics2D = scaledImage.createGraphics();
            graphics2D.drawImage(imageToScale, 0, 0, scale, scale, null);
            graphics2D.dispose();
        }
        return scaledImage;
    }
}
