import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Mlp extends IModelBase{

        int alphabetSize;
        int embeddingVectorSize;

        int contextLength;
        double epsilon;
        double momentum;
        int[] hiddenLayersNeurons;
        int outputSize;
        public List<BlockOfSequentialLayers> topo;
        boolean setToTrainingMode=false;

        double [] lossi;

        int batchSize;
        Dataset ds;

        IEncoder encoder;



    public Mlp(int embeddingVectorSize, int inputSize, int outputSize, int[] hiddenLayersNeurons, int contextLength, int alphabetSize, double epsilon, double momentum, String alphabet,TokenizerFactory.Enc encoder){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenLayersNeurons=hiddenLayersNeurons;
        this.outputSize=outputSize;
        topologyBuilder();
        this.encoder=TokenizerFactory.create(encoder,  alphabet);
        this.contextLength=contextLength;

    }
    @Override
    public List <BlockOfSequentialLayers> topologyBuilder (){
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

        this.topo=parent.buildTopo();
        return this.topo;
    }

    int epochsLasted=0;

    @Override
    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method, List<String> inputs, List<String> targets){
        this.batchSize=batchSize;
        this.ds=ds;
        IOptimizer opt=OptimizerFactory.create(descent,method);
        Tensor loss=null;
        lossi=new double[epochs];
        this.setToTrainingMode=true;
        long kAverage=0;

        int iteration=0;

        int epochsLasted=0;

        for (int i=0;i<epochs;i++){
            long startTime = System.currentTimeMillis();



            int startIndex=iteration*batchSize;
            if (startIndex+batchSize>inputs.size()){
                iteration=0;
                startIndex=0;
                epochsLasted++;
            }
            else {
                iteration=i;
            }


            List<String> batch=ds.giveMeBatch(batchSize,startIndex,Dataset.setType.TRAIN); //r:context, c:batch
            double [][] inputsEncoded=new double[batchSize][contextLength];
            double [][] targetsEncoded=new double[batchSize][1];
            //row = batch, column= context
            for (int b=0;b<batch.size();b++){
                inputsEncoded[b]=encoder.encode(inputs.get(b).toCharArray());
                targetsEncoded[b]=encoder.encode(targets.get(b).toCharArray());
            }

            //row = context, row= batch
            double [][] inputsTransposed=MathHelper.transp(inputsEncoded);
            double [][] targetsTransposed=MathHelper.transp(targetsEncoded);

            //int contextLength=aInputs.length;
            Tensor [] inputData=new Tensor[contextLength];
            for (int c=0;c<contextLength;c++){
                inputData[c]=new Tensor(batchSize, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(inputsTransposed[c]);

            }
            //Tensor inputData=new Tensor(aInputs,new HashSet<>(),"X");
            Tensor labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y").oneHot(targetsTransposed[0]);
            Tensor[] out=call(inputData);
            loss=out[0].categoricalEntropyLoss(labels);
            lossi[i]=loss.data[0][0];
            loss.backward();
            updateParameters(descent,opt,i);
            resetGradients();
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
        setToTrainingMode=false;
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
                Tensor P=O[0].softMax();
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

        for (int i=0;i<nBatches;i++){
            int startIndex=i*batchSize;
            if (i*batchSize+batchSize>setSize){
                batchSize=(i*nBatches+batchSize)-setSize;
            }
            List<String> batch=ds.giveMeBatch(batchSize,startIndex,type);
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
                inputData[c]=new Tensor(inputsTransposed[0].length, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(inputsTransposed[c]);

            }

            //Tensor inputData=new Tensor(aInputs,new HashSet<>(),"X");
            Tensor labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y").oneHot(targetsTransposed[0]);
            Tensor[] out=this.call(inputData);
            loss=out[0].categoricalEntropyLoss(labels);
            lossSum+=loss.data[0][0];

        }
        return lossSum/nBatches;
    }



}
