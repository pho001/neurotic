public class AdamOptimizer implements IOptimizer{
    // Parametry algoritmu Adam
    private double beta1;
    private double beta2;
    private double learningRate;
    private double epsilon;


    // Časový krok

    public AdamOptimizer(double beta1, double beta2, double learningRate, double epsilon){
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.learningRate = learningRate;
        this.epsilon = epsilon;


    }

    @Override
    public void update(Tensor parameter,int timeStep) {

        int rows= parameter.rows;
        int cols= parameter.cols;


        //let's initalize means and weights
        if (parameter.means==null){
            parameter.means=new double [rows][cols];
            parameter.variances=new double [rows][cols];
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                parameter.means[i][j] = beta1 * parameter.means[i][j] + (1 - beta1) * parameter.gradients[i][j];
                parameter.variances[i][j] = beta2 * parameter.variances[i][j] + (1 - beta2) * Math.pow(parameter.gradients[i][j], 2);

                //correction
                double mHat = parameter.means[i][j] / (1 - Math.pow(beta1, timeStep+1));         //timeStep can't start at 0, denominator would be 0
                double vHat = parameter.variances[i][j] / (1 - Math.pow(beta2, timeStep+1));

                // Update parameters
                parameter.data[i][j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);

            }
        }
    }
}
