import java.util.HashSet;

public class NN {

    Network N;
    Normalizer Normalizer=new Normalizer();
    public NN(int nInputs, int[] layers, String activation){
        this.N= new Network (nInputs, layers,activation);

    }
    public void train(TrainingData td, double e){


        double[][]x=td.trainingData;
        double[][] yexpected= td.expectedoutputs;
        double[][] control_data=td.control_data;
        double [][]normalized_exp= this.Normalizer.minMaxNormalization(yexpected);

        double error=Double.MAX_VALUE;
        int step=0;
        while (error>e || step<1000){
            //forward pass for all training data
            Value[][] output =N.call(x);

            Value loss=calculateMSE(normalized_exp,output);
            resetGradients();

            //back propagation
            loss.backward();
            error=loss.data;
            System.out.println(error);

            //weight update
            updateParameters(step);
            step++;
        }

    }

    private static Value calculateMSE(double[][] yTrue, Value[][] yPred) {
        if (yTrue.length != yPred.length) {
            throw new IllegalArgumentException("Expected set of values is not same as predicted set");
        }
        if (yTrue[0].length != yPred[0].length) {
            throw new IllegalArgumentException("Output sets don't match between expected and predicted sets");
        }
        int samples=yTrue.length;
        int outputs=yTrue[0].length;


        Value [][] expected= new Value [samples][outputs];
        Value sum = new Value(0, new HashSet<>(),"","Loss");
        Value scale = new Value(1/(samples*outputs), new HashSet<>(),"","Scale");



        for (int i=0;i<samples;i++) {
            for (int j=0;j<outputs;j++) {
                expected[i][j]=(new Value(yTrue[i][j], new HashSet<>(),"","Expected value " +i+"th sample, "+j+"th value"));
                Value error=expected[i][j].sub(yPred[i][j]);
                Value squaredError=error.pow(2);
                sum=sum.add(squaredError);
            }
        }
        sum.mul(scale);

        return sum;
    }

    private void updateParameters(int step){
        double learning_rate=0.01;
        //double learning_rate=0.1*Math.pow(0.9,step);
        for (int i=0;i<this.N.parameters().size();i++)
        {
            N.parameters().get(i).data+=-learning_rate*N.parameters().get(i).gradient;
        }
    }

    public void resetGradients(){
        for (int i=0;i<this.N.parameters().size();i++){
            this.N.parameters().get(i).gradient=0;
        }
    }
}
