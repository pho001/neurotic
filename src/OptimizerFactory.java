public class OptimizerFactory {
    enum Opt{
            DESC,           //gradient descent
            ADAM            //ADAM optimizer
    }
    public static IOptimizer create(double learningRate, Opt optimizer){

        switch (optimizer) {
            case DESC -> {
                return new DescentOptimizer(learningRate);
            }
            case ADAM -> {
                return new AdamOptimizer(0.9, 0.999, 0.01, 1e-8);
            }
            default -> {
                throw new RuntimeException( "Wrong Optimalization parameter selected");
            }
        }
    }
}
