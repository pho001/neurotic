import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Wavenet extends IModelBase{

    int alphabetSize;
    int embeddingVectorSize;

    int contextLength;
    double epsilon;
    double momentum;
    int hiddenLayersNeurons;
    int outputSize;
    int inputSize;
    int groupSize;

    public List<BlockOfSequentialLayers> topo;
    boolean setToTrainingMode=false;

    double [] lossi;

    int batchSize;
    Dataset ds;
    IEncoder encoder;

    public Wavenet(int embeddingVectorSize, int inputSize, int contextLength, int groupSize, int outputSize, int hiddenNeurons, int alphabetSize, double epsilon, double momentum, String alphabet,TokenizerFactory.Enc encoder){
        this.alphabetSize=alphabetSize;
        this.embeddingVectorSize=embeddingVectorSize;
        this.contextLength=contextLength;
        this.momentum=momentum;
        this.epsilon=epsilon;
        this.hiddenLayersNeurons=hiddenNeurons;
        this.outputSize=outputSize;
        this.inputSize=inputSize;
        this.groupSize=groupSize;
        topologyBuilder();
        this.encoder=TokenizerFactory.create(encoder,  alphabet);

        this.contextLength=contextLength;

        //this.topo=topology();
    }
    @Override
    public List <BlockOfSequentialLayers> topologyBuilder (){
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
        this.topo=parent.buildTopo();
        return this.topo;
    }

    @Override
    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method, List<String> inputs, List<String> targets){
        this.batchSize=batchSize;
        this.ds=ds;
        IOptimizer opt=OptimizerFactory.create(descent,method);
        Tensor loss=null;
        lossi=new double[epochs];
        this.setToTrainingMode=true;
        long kAverage=0;

        int contextLength=0;


        for (int i=0;i<epochs;i++){
            long startTime = System.currentTimeMillis();
            //TODO: i don't like this part, dataset preparation has to be fixed


            int startIndex=i*batchSize;

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

            //int contextLength=aInputs.length;
            Tensor [] inputData=new Tensor[contextLength];
            for (int c=0;c<contextLength;c++){
                inputData[c]=new Tensor(targetsTransposed.length, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(targetsTransposed[c]);

            }
            //Tensor inputData=new Tensor(aInputs,new HashSet<>(),"X");
            Tensor labels=new Tensor(batchSize,ds.getAlphabetSize(),new HashSet<>(),"Y").oneHot(targetsTransposed[0]);

            Tensor[] out=this.call(inputData);
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
                inputData[c]=new Tensor(batchSize, ds.getAlphabetSize(), new HashSet<>(),"X"+i).oneHot(inputsTransposed[c]);

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
