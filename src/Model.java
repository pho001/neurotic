import javax.sql.DataSource;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

public class Model {
    List <BlockOfSequentialLayers> topology;
    int blocksCount;
    NNConfiguration cfg;
    boolean setTrainingMode=true;
    architecture arch;
    double[] lossi;
    int contextSize;
    int batchSize;
    Dataset ds;
    public enum architecture {
            MLP,
            WAVENET
    }

    public Model(int EmbeddingVectorSize, int inputSize, int outputSize, int[] hiddenLayersNeurons, int context, architecture arch, Dataset ds, NNConfiguration cfg){
        this.cfg=cfg;
        this.arch=arch;
        this.contextSize=context;

        this.ds=ds;
        switch (arch) {
            case MLP -> {
                //this.topology=mlp(EmbeddingVectorSize,inputSize, outputSize,hiddenLayersNeurons,context,ds);
                this.topology=new Mlp(EmbeddingVectorSize,inputSize, outputSize,hiddenLayersNeurons,context,ds.getAlphabetSize(),cfg.epsilon,cfg.momentum).topology();
            }
            case WAVENET -> {
                //TODO:
                //for WAVENET, just first amount of neurons is taken into account. Needs to be fixed;
                //Also - groupSize should be hyperparameter. Needs to be fixed.
                //this.topology=wavenet(EmbeddingVectorSize,inputSize, 2,outputSize,hiddenLayersNeurons[0],ds);

                this.topology=new Wavenet(EmbeddingVectorSize,inputSize,context,2,ds.getAlphabetSize(),hiddenLayersNeurons[0],ds.getAlphabetSize(),cfg.epsilon,cfg.momentum).topology();
            }
        }
        HashSet<Tensor> parameters = parameters();


    }


    public Tensor [] call (Tensor[] input){

        for (BlockOfSequentialLayers block:this.topology){
            block.setTrainingMode=this.setTrainingMode;
            input=block.call(input);
        }
        return input;
    }

    public HashSet<Tensor> parameters () {
        HashSet <Tensor> params=new HashSet<>();
        for (BlockOfSequentialLayers block:this.topology){
            params.addAll(block.parameters());
        }
        return params;
    }


    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }


    public void train(int epochs, double descent, int batchSize,Dataset ds,int displayFrequency, OptimizerFactory.Opt method){
        this.batchSize=batchSize;
        IOptimizer opt=OptimizerFactory.create(descent,method);
        Tensor loss=null;
        lossi=new double[epochs];
        this.setTrainingMode=true;
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
            switch (arch) {
                case MLP, WAVENET -> {
                    loss=out[0].categoricalEntropyLoss(labels);
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
                System.out.println("Epoch " + (int) (i + cfg.getEpochsLasted()) + " Loss: " + loss.data[0][0] + " | last batch time: " + millis + " ms | "+i+"/"+epochs+" average: "+kAverage+" ms");
            }

        }

    }

    private void resetGradients(){
        HashSet<Tensor> parameters = parameters();
        for (Tensor p:parameters){
            p.gradients=new double[p.rows][p.cols];
        }
    }

    private void updateParameters(double descent,IOptimizer opt,int epoch){
        HashSet<Tensor> parameters = parameters();

        for (Tensor p:parameters){
            opt.update(p,epoch);
            /*
            for (int i = 0; i < p.data.length; i++) {
                for (int j = 0; j < p.data[0].length; j++) {
                    p.data[i][j] += -descent * p.gradients[i][j];

                }
            }

             */
        }
    }

    public int getParametersCount(){
        HashSet<Tensor> parameters = parameters();

        int sum=0;
        for (Tensor p:parameters){
            System.out.println(p.label+": "+p.rows+"x"+p.cols);
            sum+=p.rows*p.cols;
        }
        return sum;
    }

    public void saveParameters(String path){
        //TODO: hyperparameters and trained parameters saved properly
        FileHandler fh=new FileHandler(path);
        fh.SaveToJson(cfg);
    }

    public void loadParameters(String path){
        //TODO: hyperparameters and trained parameters loaded properly
        FileHandler fh=new FileHandler(path);
        this.cfg=fh.readFromJson();
    }

    public void generate(int nSamples) {
        this.setTrainingMode = false;

        switch (this.arch) {

            case MLP, WAVENET -> {
                generateText(nSamples);
            }

        }


    }

    public void displayGraph(int reducedRow){
        double [][] t=new double [][] {this.lossi};
        double [] lossi_new=MathHelper.calculateAverages(this.lossi,reducedRow);
        Chart chart = new Chart("Losses Over Time", "Epoch*"+reducedRow, "Loss");
        chart.addSeries(lossi_new);
        chart.display();
    }

    public void predict(double[] inputs){
        //TODO: implement numerical prediction
    }

    private void generateText(int nSamples){
        for (int i = 0; i < nSamples; i++) {
            String output = "";
            String context = "";
            for (int j = 0; j < this.contextSize; j++) {
                context += ".";
            }
            Random random=new Random();
            while (true) {
                double[][] cArray = new double[contextSize][1];
                Tensor [] X=new Tensor[contextSize];
                //Tensor [] X=new Tensor (cArray,new HashSet<>(),"X");
                for (int k = 0; k < context.length(); k++) {
                    cArray[k][0] = (double)this.ds.strtoi(context.charAt(k));
                    X[k]=new Tensor (1, ds.getAlphabetSize(), new HashSet<>(),"X").oneHot(new double [] {this.ds.strtoi(context.charAt(k))});
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
                for (int k=0;k<contextSize-1;k++){
                    tmp+=context.charAt((k+1));
                }
                context=tmp;
                context+=ds.itostr(vector);
            }
            System.out.println(output);

        }

    }
    public double splitLoss(Dataset.setType type){
        Tensor loss=null;
        double lossSum=0;
        this.setTrainingMode = false;
        int setSize=this.ds.giveMeSet(type)[0].length;
        int nBatches=(int)(Math.floor((double)setSize/this.batchSize));






        //TODO: needs to be fixed. Last batch, which is smaller then batchSize is ignored.

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
            switch (arch) {
                case MLP, WAVENET -> {
                    lossSum+=out[0].categoricalEntropyLoss(labels).data[0][0];
                }
            }

        }
        return lossSum/nBatches;
    }




}
