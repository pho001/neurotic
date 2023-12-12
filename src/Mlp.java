import java.util.List;
import java.util.Set;

public class Mlp implements IModel{

        int alphabetSize;
        int embeddingVectorSize;

        int contextLength;
        double epsilon;
        double momentum;
        int[] hiddenLayersNeurons;
        int outputSize;

    public Mlp(int embeddingVectorSize, int inputSize, int outputSize, int[] hiddenLayersNeurons, int contextLength, int alphabetSize, double epsilon, double momentum){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenLayersNeurons=hiddenLayersNeurons;
        this.outputSize=outputSize;
        //this.topo=topology();
    }
    @Override
    public List <BlockOfSequentialLayers> topology (){
        //Embedding layer

        BlockOfSequentialLayers parent=new BlockOfSequentialLayers(List.of(
                new EmbeddingLayer(alphabetSize,embeddingVectorSize, contextLength),
                new FlattenLayer(contextLength)
        ), "Embedding layer");
        int prevLayerNeurons=embeddingVectorSize*contextLength;

        //construct hidden layers
        for (int i=0;i<hiddenLayersNeurons.length;i++){

            int neurons=hiddenLayersNeurons[i];
            if (i==0){
                parent = new BlockOfSequentialLayers(List.of(
                        new LinearLayer(prevLayerNeurons, neurons,true),
                        new BatchNormLayer(neurons, epsilon, momentum),
                        new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
                ), "BatchNorm layer " + i).setParent(Set.of(parent));
            }
            else {
                parent = new BlockOfSequentialLayers(List.of(
                        new LinearLayer(prevLayerNeurons, neurons, true),
                        new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
                ), "Hidden layer " + i).setParent(Set.of(parent));
            }
            prevLayerNeurons=neurons;
        }

        //construct output layer
        parent = new BlockOfSequentialLayers(List.of(
                new LinearLayer(prevLayerNeurons,outputSize,true)
        ), "Output layer").setParent(Set.of(parent));

        return parent.buildTopo();
    }



}
