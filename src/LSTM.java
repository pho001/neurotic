import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class LSTM extends IModelBase{
    int alphabetSize;
    int embeddingVectorSize;

    int contextLength;
    double epsilon;
    double momentum;

    int outputSize;
    int inputSize;
    public List<BlockOfSequentialLayers> topo;
    boolean setToTrainingMode=false;
    int [] hiddenStates;
    int inputNeurons;

    double [] lossi;

    int batchSize;
    Dataset ds;
    IEncoder encoder;


    public LSTM(int embeddingVectorSize, int inputSize, int outputSize, int inputNeurons,int[] hiddenStates, int contextLength, int alphabetSize, double epsilon, double momentum, String alphabet,TokenizerFactory.Enc encoder){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenStates=hiddenStates;
        this.outputSize=outputSize;
        this.inputSize=inputSize;
        this.inputNeurons=inputNeurons;
        topologyBuilder();
        this.encoder=TokenizerFactory.create(encoder,  alphabet);

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
        for (int j = 0; j < hiddenStates.length; j++){
            if (j == 0) {
                parent = new BlockOfSequentialLayers(List.of(
                        //new LinearLayer(prevLayerNeurons, inputNeurons, true),
                        new OneDirectionalLSTMLayer(true, hiddenStates[j])
                        //new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)

                ), "Hidden layer : "+j).setParent(Set.of(parent));
            }
            else {
                parent = new BlockOfSequentialLayers(List.of(
                        new OneDirectionalLSTMLayer(true,hiddenStates[j])
                        //new NonLinearLayer(NonLinearLayer.Nonlinearity.TANH)
                ), "Hidden layer : "+j).setParent(Set.of(parent));
            }

            prevLayerNeurons=hiddenStates[j];
            if(j== hiddenStates.length-1){
                parent = new BlockOfSequentialLayers(List.of(
                        new LinearLayer(prevLayerNeurons, outputSize, true)
                ), "Output layer").setParent(Set.of(parent));
            }
        }
        this.topo=parent.buildTopo();
        return this.topo;

    }




    @Override
    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method,List<String> inputs,List<String> targets){
        this.batchSize=batchSize;
        this.ds=ds;
        IOptimizer opt=OptimizerFactory.create(descent,method);


        Tensor loss=null;
        lossi=new double[epochs];
        this.setToTrainingMode=true;
        long kAverage=0;

        int contextLength=0;
        int epochsLasted=0;

        for (int i=0;i<epochs;i++){
            long startTime = System.currentTimeMillis();
            int startIndex=i*batchSize;
            //TODO: i don't like this part, dataset preparation has to be fixed
            List<String> batch=ds.giveMeBatch(batchSize,startIndex,Dataset.setType.TRAIN); //r:context, c:batch
            double [][] inputsEncoded=null;
            double [][] targetsEncoded=null;
            //row = batch, column= context
            for (int b=0;b<batch.size();b++){
                inputsEncoded[b]=encoder.encode(batch.get(b).toCharArray());
                targetsEncoded[b]=encoder.encode(batch.get(b).toCharArray());
            }

            //row = context, row= batch
            double [][] inputsTransposed=MathHelper.transp(inputsEncoded);
            double [][] targetsTransposed=MathHelper.transp(inputsEncoded);

            Tensor [] inputData=new Tensor[contextLength];
            for (int c=0;c<contextLength;c++){
                inputData[c]=new Tensor(batchSize, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(inputsTransposed[c]);

            }

            double [] nextInput=inputsTransposed[0];
            for (int t=0;t<contextLength;t++){
                inputData[0]=new Tensor(inputsTransposed[0].length, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(nextInput);
                Tensor target;
                if (t==contextLength-1) {
                    target = new Tensor(batchSize, ds.getAlphabetSize(), new HashSet<>(), "Y" + t).oneHot(targetsEncoded[0]);
                }
                else {
                    target = new Tensor(batchSize, ds.getAlphabetSize(), new HashSet<>(), "Y" + t).oneHot(targetsEncoded[t+1]);
                }

                Tensor out[]=call(inputData);
                Tensor P=out[0].softMax();
                double [] outChar=new double[P.rows];
                for (int p=0;p<P.rows;p++){
                    int maxProbIndex = MathHelper.findMaxIndex(P.data[p]);
                    if (MathHelper.countOccurrences(P.data[p], P.data[p][maxProbIndex])>1){
                        Random random = new Random();
                        nextInput[p] = MathHelper.generateMultinomialVector(1,P.data[p],random)[0];
                    }
                    else {
                        nextInput[p]=maxProbIndex;
                    }

                }

                if (t==0){
                    loss=out[0].categoricalEntropyLoss(target);
                }
                else{
                    loss=loss.add(out[0].categoricalEntropyLoss(target));
                }
            }
            lossi[i]=loss.data[0][0];
            loss.backward();

            this.updateParameters(descent,opt,i);
            this.resetGradients();
            long endTime = System.currentTimeMillis();
            long millis = endTime-startTime;
            if (displayFrequency>0 && i%displayFrequency==0){
                kAverage+=millis;
                System.out.println("Epoch " + (int) (i) + " Loss: " + lossi[i] + " | last batch time: " + millis + " ms | "+i+"/"+epochs+" average: "+kAverage+" ms");
            }

        }
    }


    @Override
    public void generate(int nSamples){
        this.setToTrainingMode = false;
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
                Tensor [] O=call(X);

                Tensor P=O[contextLength-1].softMax();
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
        int setSize=this.ds.giveMeSet(type).size();
        int nBatches=(int)(Math.floor((double)setSize/this.batchSize));
        //TODO: needs to be fixed. Last batch, which is smaller then batchSize is ignored. Data preparation needs to be moved out of this class

        for (int i=0;i<nBatches;i++) {
            int startIndex = i * batchSize;
            if (i * batchSize + batchSize > setSize) {
                batchSize = (i * nBatches + batchSize) - setSize;
            }
            List<String> batch = ds.giveMeBatch(batchSize, startIndex, Dataset.setType.TRAIN); //r:context, c:batch
            double[][] inputsEncoded = null;
            double[][] targetsEncoded = null;
            //row = batch, column= context
            for (int b = 0; b < batch.size(); b++) {
                inputsEncoded[b] = encoder.encode(batch.get(b).toCharArray());
                targetsEncoded[b] = encoder.encode(batch.get(b).toCharArray());
            }

            //row = context, row= batch
            double[][] inputsTransposed = MathHelper.transp(inputsEncoded);
            double[][] targetsTransposed = MathHelper.transp(inputsEncoded);
        }
        return lossSum/nBatches;
    }


}
