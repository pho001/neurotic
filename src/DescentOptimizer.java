public class DescentOptimizer implements  IOptimizer{

    double learningRate;
    public DescentOptimizer(double learningRate){
        this.learningRate=learningRate;
    }

    @Override
    public void update(Tensor Parameter,int epoch) {
        for (int i=0;i<Parameter.rows;i++){
            for(int j=0;j<Parameter.cols;j++){
                Parameter.data[i][j]+=-learningRate*Parameter.gradients[i][j];
            }
        }
    }
}
