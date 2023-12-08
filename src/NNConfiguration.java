import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class NNConfiguration {
    int contextSize=10;
    int neuronsHid=300;
    double descent=0.1;
    double train=0.8;
    double test=0.1;
    double dev=0.1;

    int epochsLasted=0;
    int batchSize=32;

    double alpha=0.999;
    int layers=2;
    int numClasses=0;

    double epsilon=1e-5;

    public double[][] means;

    public double[][] vars;

    double momentum=0.999;

    //Map<String, double [][]> parameters=new HashMap<>();
    HashMap<String, double[][]> parameters = new HashMap<>();
    //HashSet<Tensor> parameters= new HashSet<>();
    public NNConfiguration(){


    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }

    public void setContextSize(int size){
        this.contextSize=size;
    }
    public int getContextSize(){
        return this.contextSize;
    }

    public int getNeuronsHid(){
        return this.neuronsHid;
    }

    public int getLayers(){
        return this.layers;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setDescent(double descent){
        this.descent=descent;
    }

    public double getDescent(){
        return this.descent;
    }
    public void setTrainDevTestRatio(double train, double dev, double test){
        this.train=train;
        this.dev=dev;
        this.test=test;
    }
    public double getTrain(){
        return this.train;
    }

    public double getTest(){
        return this.test;
    }
    public double getDev(){
        return this.dev;
    }

    public int getEpochsLasted(){
        return this.epochsLasted;
    }

    public void setEpochsLasted(int epochs){
        this.epochsLasted+=epochs;
    }
    public void setBatchSize(int size){
        this.batchSize=size;
    }

    public int getBatchSize(){
        return this.batchSize;
    }

    public void setNumClasses(int numClasses){
        this.numClasses=numClasses;
    }

    public int getNumClasses(){
        return this.numClasses;
    }

    public double[][] getMeans() {return this.means;}

    public double[][] getVars() {return this.vars;}

    public void setVars(double [][] vars) {this.vars=vars;}
    public void setMeans(double [][] means) {this.means=means;}



    public HashMap<String,double[][]> getParameters(){
        return this.parameters;
    }

    public void setParameters(HashMap<String,double[][]> parameters){
        this.parameters=parameters;
    }

    public void storeParameters(HashSet<Tensor> parameters){
        HashMap<String,double[][]> out=new HashMap<>();
        for (Tensor p:parameters) {
            out.put(p.label,p.data);
        }
        this.parameters=out;
    }





    /*
    public void initParameters(int[] layers) {
        int previousLayer=0;
        int currentLayer=0;
        for (int i=0;i<layers.length;i++){

            if (i == 0) {
                this.parameters.put("W"+i,new double[this.numClasses][layers[i]]);
                this.parameters.put("B"+i,new double[batchSize][layers[i]]);
            }
            else{
                currentLayer=layers[i];
                previousLayer=layers[i-1];
                this.parameters.put("W"+i,new double[previousLayer][currentLayer]);
                this.parameters.put("B"+i,new double[batchSize][layers[i]]);
            }

        }
    }
    */




}
