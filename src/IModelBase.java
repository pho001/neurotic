import java.util.HashSet;
import java.util.List;

abstract public class IModelBase implements IModel{
    protected abstract List<BlockOfSequentialLayers> getTopo();
    protected abstract double[] getLosses();

    protected abstract boolean isInTrainingMode();

    protected void resetGradients(){
        HashSet<Tensor> parameters = getParameters();
        for (Tensor p:parameters){
            p.gradients=new double[p.rows][p.cols];
        }
    }

    protected void updateParameters(double descent,IOptimizer opt,int epoch){
        HashSet<Tensor> parameters = getParameters();
        for (Tensor p:parameters){
            opt.update(p,epoch+1);

        }
    }
    public HashSet<Tensor> getParameters () {
        HashSet <Tensor> params=new HashSet<>();
        for (BlockOfSequentialLayers block:getTopo()){
            params.addAll(block.parameters());
        }
        return params;
    }

    @Override
    public void displayGraph(int reducedRow){
        double [] lossi_new=MathHelper.calculateAverages(getLosses(),reducedRow);
        Chart chart = new Chart("Losses Over Time", "Epoch*"+reducedRow, "Loss");
        chart.addSeries(lossi_new);
        chart.display();
    }

    @Override
    public int getParametersCount(){
        HashSet <Tensor> params=getParameters();
        int sum=0;
        for (Tensor T:params){
            sum+=T.rows*T.cols;
        }
        return sum;
    }

    @Override
    public Tensor [] call (Tensor[] input){

        for (BlockOfSequentialLayers block:getTopo()){
            block.setTrainingMode=isInTrainingMode();
            input=block.call(input);
        }
        return input;
    }



}
