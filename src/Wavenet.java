import java.util.List;
import java.util.Set;

public class Wavenet implements IModel{

    int alphabetSize;
    int embeddingVectorSize;

    int contextLength;
    double epsilon;
    double momentum;
    int hiddenLayersNeurons;
    int outputSize;
    int inputSize;
    int groupSize;



    public Wavenet(int embeddingVectorSize, int inputSize, int contextLength, int groupSize, int outputSize, int hiddenNeurons, int alphabetSize, double epsilon, double momentum){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenLayersNeurons=hiddenNeurons;
        this.outputSize=outputSize;
        this.inputSize=inputSize;
        this.groupSize=groupSize;

        //this.topo=topology();
    }
    @Override
    public List <BlockOfSequentialLayers> topology (){
        //Embedding layer

        //transform input Tensor X (context,batch) -> Tensor [] X (context, batch, vocabsize)

        //----->Tensor X (context,batch)
        BlockOfSequentialLayers parent = new BlockOfSequentialLayers(List.of(
                new EmbeddingLayer(inputSize,embeddingVectorSize,contextLength)
        ),"Embedding Layer");

        //----->Tensor [] X (context,batch, vocabsize)

        int prevLayerNeurons=embeddingVectorSize;

        int levels=(int) (Math.log(contextLength) / Math.log(groupSize));

        for (int i=0;i<levels;i++){
            if (i<levels){
                prevLayerNeurons=prevLayerNeurons*groupSize;
                hiddenLayersNeurons=hiddenLayersNeurons;    //meh
            }
            else {
                hiddenLayersNeurons=hiddenLayersNeurons;   //last layer is output layer
            }



            //prevLayerNeurons=(i+1)*embeddingVectorSize*groupSize;
            parent = new BlockOfSequentialLayers(List.of(
                    new FlattenLayer(groupSize),
                    //----->Tensor[] X(context/groupsize,batchSize,vocabsize)
                    new LinearLayer(prevLayerNeurons, hiddenLayersNeurons,false),
                    //----->Tensor[] X(context/groupsize,prevLatyerNerurons,Neurons)
                    new BatchNormLayer(hiddenLayersNeurons,epsilon,momentum),
                    //----->Tensor[] X(context/groupsize,prevLatyerNerurons,Neurons)
                    new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
            ),"Concated linear layers").setParent(Set.of(parent));
            prevLayerNeurons=hiddenLayersNeurons;
        }
        //construct output layer
        parent = new BlockOfSequentialLayers(List.of(
                new LinearLayer(prevLayerNeurons,alphabetSize,true)
        ), "Output layer").setParent(Set.of(parent));
        return parent.buildTopo();
    }



}
