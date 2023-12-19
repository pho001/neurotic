import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Rnn extends IModelBase{
    int alphabetSize;
    int embeddingVectorSize;

    int contextLength;
    double epsilon;
    double momentum;
    int[] hiddenLayersNeurons;
    int outputSize;
    int inputSize;
    public List<BlockOfSequentialLayers> topo;
    boolean setToTrainingMode=false;

    double [] lossi;

    int batchSize;
    Dataset ds;


    public Rnn(int embeddingVectorSize, int inputSize, int outputSize, int[] hiddenLayersNeurons, int contextLength, int alphabetSize, double epsilon, double momentum){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenLayersNeurons=hiddenLayersNeurons;
        this.outputSize=outputSize;
        this.inputSize=inputSize;
        topologyBuilder();

    }

    @Override
    public List<BlockOfSequentialLayers> topologyBuilder() {
        //Embedding layer
        BlockOfSequentialLayers parent = new BlockOfSequentialLayers(List.of(
                new EmbeddingLayer(inputSize, embeddingVectorSize, contextLength)
                //new BatchNormLayer(100,epsilon,momentum)
        ), "Embedding Layer");
        int prevLayerNeurons = embeddingVectorSize;

        //Recurrent layers construction
        for (int j = 0; j < hiddenLayersNeurons.length; j++){
            if (j == 0) {
                parent = new BlockOfSequentialLayers(List.of(
                        new LinearLayer(prevLayerNeurons, hiddenLayersNeurons[j], false),
                        new OneDirectionalRNNLayer(hiddenLayersNeurons[j], false),
                        new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
                ), "Hidden layer : "+j).setParent(Set.of(parent));
            }
            else {
                parent = new BlockOfSequentialLayers(List.of(
                        new OneDirectionalRNNLayer(hiddenLayersNeurons[j], false),
                        new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
                ), "Hidden layer : "+j).setParent(Set.of(parent));
            }

            prevLayerNeurons=hiddenLayersNeurons[j];
            if(j== hiddenLayersNeurons.length-1){
                parent = new BlockOfSequentialLayers(List.of(
                        new LinearLayer(prevLayerNeurons, outputSize, true)
                ), "Output layer").setParent(Set.of(parent));
            }
        }
        this.topo=parent.buildTopo();
        return this.topo;

    }




    @Override
    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method){
        this.batchSize=batchSize;
        this.ds=ds;
        IOptimizer opt=OptimizerFactory.create(descent,method);
        Tensor loss=null;
        lossi=new double[epochs];
        this.setToTrainingMode=true;
        long kAverage=0;
        if (ds.trainSet.length<=1){
            throw new RuntimeException("Training data dimension is too low, it doesn't contain labels");
        }
        int contextLength=0;


        for (int i=0;i<epochs;i++){
            long startTime = System.currentTimeMillis();
            //TODO: i don't like this part, dataset preparation has to be fixed
            double[][] batch=ds.giveMeRandomBatch(Dataset.setType.TRAIN,batchSize); //r:context, c:batch
            contextLength=batch.length-1;
            double [][] aLabels=new double [][]{batch [contextLength]};
            double [][] aInputs=new double[contextLength][];
            for (int j=0;j<contextLength;j++){
                aInputs[j]=batch[j];
            }
            //int contextLength=aInputs.length;
            Tensor [] inputData=new Tensor[contextLength];
            for (int c=0;c<contextLength;c++){
                inputData[c]=new Tensor(aInputs[0].length, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(aInputs[c]);

            }
            //Tensor inputData=new Tensor(aInputs,new HashSet<>(),"X");
            Tensor labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y").oneHot(aLabels[0]);
            Tensor[] out=this.call(inputData);

            for (int o=0;o<out.length;o++){
                labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y"+o).oneHot(batch[o]);
                if (o==0){
                    loss=out[o].categoricalEntropyLoss(labels);
                }
                else{
                    loss=loss.add(out[o].categoricalEntropyLoss(labels));
                }
                lossi[i]=loss.data[0][0];

            }
            loss.backward();
            this.updateParameters(descent,opt,i);
            this.resetGradients();
            long endTime = System.currentTimeMillis();
            long millis = endTime-startTime;
            if (displayFrequency>0 && i%displayFrequency==0){
                kAverage+=millis;
                System.out.println("Epoch " + (int) (i) + " Loss: " + loss.data[0][0] + " | last batch time: " + millis + " ms | "+i+"/"+epochs+" average: "+kAverage+" ms");
            }

        }

    }


    @Override
    public void generate(int nSamples){
        for (int i = 0; i < nSamples; i++) {
            String output = "";
            String context = "";
            for (int j = 0; j < this.contextLength; j++) {
                context += ".";
            }
            Random random=new Random();
            while (true) {
                double[][] cArray = new double[contextLength][1];
                Tensor [] X=new Tensor[contextLength];
                //Tensor [] X=new Tensor (cArray,new HashSet<>(),"X");
                for (int k = 0; k < context.length(); k++) {
                    cArray[k][0] = (double)this.ds.strtoi(context.charAt(k));
                    X[k]=new Tensor (1, alphabetSize, new HashSet<>(),"X").oneHot(new double [] {this.ds.strtoi(context.charAt(k))});
                    //X[i].data[0][0]=(double)this.ds.strtoi(context.charAt(k));

                }
                //Tensor X=new Tensor (cArray,new HashSet<>(),"X");
                Tensor P=this.call(X)[0].softMax();
                int[] vector= MathHelper.generateMultinomialVector(1, P.data[0], random);
                output=output+ds.itostr(vector);
                if (vector[0]==0){
                    break;
                }
                String tmp="";
                for (int k=0;k<contextLength-1;k++){
                    tmp+=context.charAt((k+1));
                }
                context=tmp;
                context+=ds.itostr(vector);
            }
            System.out.println(output);

        }

    }

    @Override
    protected List<BlockOfSequentialLayers> getTopo(){
        return this.topo;
    }

    @Override
    protected double[] getLosses(){
        return this.lossi;
    }

    public void predict(double[] inputs){

    }


    public void saveParameters(String path){
        //TODO: hyperparameters and trained parameters saved properly
    }

    public void loadParameters(String path){
        //TODO: hyperparameters and trained parameters loaded properly
    }

    @Override
    public boolean isInTrainingMode(){
        return setToTrainingMode;
    }

    @Override
    public double splitLoss(Dataset.setType type){
        Tensor loss=null;
        double lossSum=0;
        this.setToTrainingMode = false;
        int setSize=this.ds.giveMeSet(type)[0].length;
        int nBatches=(int)(Math.floor((double)setSize/this.batchSize));
        //TODO: needs to be fixed. Last batch, which is smaller then batchSize is ignored. Data preparation needs to be moved out of this class

        for (int i=0;i<nBatches;i++){
            int startIndex=i*batchSize;
            if (i*batchSize+batchSize>setSize){
                batchSize=(i*nBatches+batchSize)-setSize;
            }
            double [][] batch=ds.giveMeBatch(batchSize,startIndex,type);
            double [][] aLabels=new double [][]{batch [batch.length-1]};
            double [][] aInputs=new double[batch.length-1][];

            for (int j=0;j<batch.length-1;j++){
                aInputs[j]=batch[j];
            }
            int contextLength=aInputs.length;
            Tensor [] inputData=new Tensor[contextLength];
            for (int c=0;c<contextLength;c++){
                inputData[c]=new Tensor(aInputs[0].length, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(aInputs[c]);

            }

            //Tensor inputData=new Tensor(aInputs,new HashSet<>(),"X");
            Tensor labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y").oneHot(aLabels[0]);
            Tensor[] out=this.call(inputData);
            for (int o=0;o<out.length;o++){
                labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y"+o).oneHot(batch[o]);
                if (o==0){
                    loss=out[o].categoricalEntropyLoss(labels);
                }
                else{
                    loss=loss.add(out[o].categoricalEntropyLoss(labels));
                }
                //lossi[i]=loss.data[0][0];
                lossSum+=loss.data[0][0];

            }

        }
        return lossSum/nBatches;
    }


}
