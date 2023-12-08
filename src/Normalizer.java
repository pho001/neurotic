public class Normalizer {
    private double min;
    private double max;
    public Normalizer(){
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;

    }

    public double[][] minMaxNormalization(double [][] data) {
        int numRows = data.length;
        int numCols = data[0].length;


        for (int i = 0; i < numRows; i++) {

            for (int j = 0; j < numCols; j++) {

                if (data[i][j] < this.min) {
                    this.min = data[i][j];
                }
                if (data[i][j] > this.max) {
                    this.max = data[i][j];
                }
            }

        }



        double[][] normalizedData = new double[numRows][numCols];
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                normalizedData[i][j] = (data[i][j] - this.min) / (this.max - this.min);
            }
        }

        return normalizedData;
    }

    public double[][] minmaxDenormalize(double[][] data) {
        int numRows = data.length;
        int numCols = data[0].length;
        double[][] denormalizedData = new double[numRows][numCols];


        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                denormalizedData[i][j] = data[i][j] * (this.max - this.min) + this.min;
            }
        }

        return denormalizedData;
    }
}
