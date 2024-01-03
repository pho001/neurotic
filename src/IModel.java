import java.util.List;

public interface IModel {

    public List <BlockOfSequentialLayers> topologyBuilder ();
    public void generate(int nSamples);
    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method);
    public void saveParameters(String path);
    public void loadParameters(String path);
    public void predict(double[] inputs);
    public void displayGraph(int reducedRow);
    public int getParametersCount();
    public Tensor [] call (Tensor[] input);
    public double splitLoss(Dataset.setType type);

}
