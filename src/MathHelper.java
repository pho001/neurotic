import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class MathHelper {
    public enum Dimension{
        BYROWS,
        BYCOLS,
        WHOLE
    }

    public static double[][] add(double[][] first, double[][] second){

        int colsFirst=first[0].length;
        int colsSecond=second[0].length;
        int rowsFirst=first.length;
        int rowsSecond=second.length;
        if (rowsFirst!=rowsSecond || colsFirst!=colsSecond){
            throw new IllegalArgumentException("Dimensions don't match; addition is not possible");
        }
        double [][] out= new double[rowsFirst][colsFirst];
        for (int i = 0; i < rowsFirst; i++) {
            for (int j = 0; j < colsSecond; j++) {

                out[i][j] = first[i][j] + second[i][j];

            }
        }
        return out;

    }

    //Addition with broadcasting of second Matrix
    public static double[][] addb(double[][] firstMatrix, double[][] secondMatrix){
        //tbd: broadcasting between other axis of second matrix. Second matrix is allways broadcasted
        int colsFirst=firstMatrix[0].length;
        int colsSecond=secondMatrix[0].length;
        int rowsFirst=firstMatrix.length;
        int rowsSecond=secondMatrix.length;
        double [][] out= new double[rowsFirst][colsFirst];
        ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        if((rowsSecond==1) && (colsFirst==colsSecond) ) {

            for (int i = 0; i < out.length; i++) {
                final int row = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out[0].length; j++) {
                        out[row][j] = firstMatrix[row][j] + secondMatrix[0][j];
                    }
                });
            }
        }
        else if((colsSecond==1) && (rowsFirst==rowsSecond)) {
            for (int i = 0; i < out[0].length; i++) {
                final int col = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out.length; j++) {
                        out[j][col] = firstMatrix[j][col] + secondMatrix[j][0];
                    }
                });
            }
        }
        else {
            throw new IllegalArgumentException("Dimensions don't satisfy conditions for broadcasting");
        }
        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static double[][] sub(double [][] firstMatrix, double [][] secondMatrix){
        double [][] out= new double[firstMatrix.length][firstMatrix[0].length];
        if (firstMatrix.length!=secondMatrix.length || firstMatrix[0].length!=secondMatrix[0].length){
            throw new IllegalArgumentException("Dimensions don't match; addition is not possible");
        }
        ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        for (int i = 0; i < out.length; i++) {
            final int row=i;
            executorService.execute(() -> {
                for (int j = 0; j < out[0].length; j++) {
                    out[row][j] = firstMatrix[row][j] - secondMatrix[row][j];
                }
            });
        }
        executorService.shutdown();

        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static double[][] subb(double [][] firstMatrix, double [][] secondMatrix){
        int colsFirst=firstMatrix[0].length;
        int colsSecond=secondMatrix[0].length;
        int rowsFirst=firstMatrix.length;
        int rowsSecond=secondMatrix.length;
        double [][] out= new double[firstMatrix.length][firstMatrix[0].length];


        ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        if((rowsSecond==1) && (colsFirst==colsSecond) ) {

            for (int i = 0; i < out.length; i++) {
                final int row = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out[0].length; j++) {
                        out[row][j] = firstMatrix[row][j] - secondMatrix[0][j];
                    }
                });
            }
        }
        else if((colsSecond==1) && (rowsFirst==rowsSecond)) {
            for (int i = 0; i < out[0].length; i++) {
                final int col = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out.length; j++) {
                        out[j][col] = firstMatrix[j][col] - secondMatrix[j][0];
                    }
                });
            }
        }
        else {
            throw new IllegalArgumentException("Dimensions don't satisfy conditions for broadcasting");
        }
        executorService.shutdown();

        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static double[][] mul(double[][] firstMatrix,double [][] secondMatrix){
        if (firstMatrix[0].length != secondMatrix.length) {
            throw new IllegalArgumentException("Dimensions don't match; multiplication is not possible");
        }

        //let's use threads top make dot products faster
        int numberOfThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        //ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        double [][] out=new double[firstMatrix.length][secondMatrix[0].length];
        for (int i = 0; i < out.length; i++) {
            final int row=i;
            executorService.execute(() ->{
                for (int j = 0; j < out[0].length; j++) {
                    out[row][j] = 0;
                    for (int k = 0; k < firstMatrix[0].length; k++) {
                        out[row][j] += firstMatrix[row][k] * secondMatrix[k][j];
                    }
                }
            });
        }
        executorService.shutdown();

        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return out;
    }

    public static double[][] divb(double [][] firstMatrix, double [][] secondMatrix){
        int colsFirst=firstMatrix[0].length;
        int colsSecond=secondMatrix[0].length;
        int rowsFirst=firstMatrix.length;
        int rowsSecond=secondMatrix.length;
        double [][] out= new double[firstMatrix.length][firstMatrix[0].length];


        ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        if((rowsSecond==1) && (colsFirst==colsSecond) ) {

            for (int i = 0; i < out.length; i++) {
                final int row = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out[0].length; j++) {
                        out[row][j] = firstMatrix[row][j] / secondMatrix[0][j];
                    }
                });
            }
        }
        else if((colsSecond==1) && (rowsFirst==rowsSecond)) {
            for (int i = 0; i < out[0].length; i++) {
                final int col = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out.length; j++) {
                        out[j][col] = firstMatrix[j][col] / secondMatrix[j][0];
                    }
                });
            }
        }
        else {
            throw new IllegalArgumentException("Dimensions don't satisfy conditions for broadcasting");
        }
        executorService.shutdown();

        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static double[][] mulb(double [][] firstMatrix, double [][] secondMatrix){
        int colsFirst=firstMatrix[0].length;
        int colsSecond=secondMatrix[0].length;
        int rowsFirst=firstMatrix.length;
        int rowsSecond=secondMatrix.length;
        double [][] out= new double[firstMatrix.length][firstMatrix[0].length];


        ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

        if((rowsSecond==1) && (colsFirst==colsSecond) ) {

            for (int i = 0; i < out.length; i++) {
                final int row = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out[0].length; j++) {
                        out[row][j] = firstMatrix[row][j] * secondMatrix[0][j];
                    }
                });
            }
        }
        else if((colsSecond==1) && (rowsFirst==rowsSecond)) {
            for (int i = 0; i < out[0].length; i++) {
                final int col = i;
                executorService.execute(() -> {
                    for (int j = 0; j < out.length; j++) {
                        out[j][col] = firstMatrix[j][col] * secondMatrix[j][0];
                    }
                });
            }
        }
        else {
            throw new IllegalArgumentException("Dimensions don't satisfy conditions for broadcasting");
        }
        executorService.shutdown();

        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }


    public static double[][] normalize(double[][] data, Tensor.Dimension dim){
        int rows=data.length;
        int cols=data[0].length;
        double [][] out=new double[rows][cols];
        double [][] mean=null;
        double [][] variance=null;
        double [][] std=null;

        switch (dim){
            case BYCOLS :
                mean=mean(data, Tensor.Dimension.BYCOLS);
                variance=variance(data,mean);

                for (int c=0;c<cols;c++) {
                    for (int r = 0; r < rows; r++) {
                        out[r][c] = (data[r][c]-mean[0][c])/Math.sqrt(variance[0][c]);
                        //out[r][c] = (data[r][c]-mean[0][c])/(std[0][c]);
                    }
                }
                break;
            case BYROWS:
                mean=mean(data, Tensor.Dimension.BYROWS);
                variance=variance(data,mean);

                for (int r=0;r<rows;r++) {
                    for (int c = 0; c < cols; c++) {
                        out[r][c] = (data[r][c]-mean[r][0])/Math.sqrt(variance[r][0]);
                        //out[r][c] = (data[r][c]-mean[r][0])/std[r][0];

                    }
                }
                break;
            case WHOLE:
                break;
        }


        return out;
    }

    public static double[][] mean(double[][] data, Tensor.Dimension dim){

        int count=0;
        int rows= data.length;
        int cols=data[0].length;

        int numberOfThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        double [][] sum=null;
        switch (dim){
            case BYROWS :
                double [][] rowssum = new double[rows][1];

                for (int r=0;r<rows;r++){
                    final int row=r;
                    executorService.execute(() -> {
                        double tmp_sum = 0.0;
                        for (int c = 0; c < cols; c++) {
                            tmp_sum += data[row][c];

                        }
                        rowssum[row][0] = tmp_sum/ cols;
                    });

                }
                sum=rowssum;
                break;
            case BYCOLS:

                double [][] colssum = new double[1][cols];

                for (int c=0;c<cols;c++){
                    final int col=c;
                    executorService.execute(() -> {
                        double tmp_sum = 0.0;
                        for (int r=0;r<rows;r++){
                            tmp_sum+=data[r][col];
                        }
                        colssum[0][col]=tmp_sum/rows;

                    });

                }
                sum=colssum;
                break;
            case WHOLE:
                sum = new double[1][1];
                for (int i=0;i<rows;i++){
                    for (int j=0;j<cols;j++){
                        sum[0][0]+=data[i][j];
                    }
                }
                sum[0][0]=sum[0][0]/(rows*cols);
                break;
        }
        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        return sum;
    }

    public static double variance(double[][] in,double mean){
        double out=0;

        int count=0;
        double sum=0;
        for (int i=0;i<in.length;i++){
            for (int j=0;j<in[0].length;j++){
                sum+=Math.pow(in[i][j]-mean,2);
                count++;
            }
        }

        out=sum/count;
        return out;
    }

    public static double [][] variance(double[][] in,double[][] means){
        double [][] out=null;
        int meansRows=means.length;
        int meansCols=means[0].length;
        int numberOfThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);


        if (meansRows==1){
            double [][] colsout=new double[1][meansCols];
            for (int c=0;c<in[0].length;c++){
                final int col=c;
                executorService.execute(() -> {
                    for (int r=0;r<in.length;r++){
                        colsout[0][col]+=Math.pow(in[r][col]-means[0][col],2);
                    }
                    colsout[0][col]=colsout[0][col]/(in.length-1);
                });

            }
            out=colsout;
        }


        else if (meansCols==1){
            double [][] rowsout=new double[meansRows][1];

            for (int r=0;r<in.length;r++){
                final int row=r;
                executorService.execute(() -> {
                    for (int c = 0; c < in[0].length; c++) {
                        rowsout[row][0] += Math.pow(in[row][c] - means[row][0], 2);
                    }
                    rowsout[row][0] = rowsout[row][0] / (in[0].length - 1);
                });

            }
            out=rowsout;
        }
        else {
            throw new IllegalArgumentException("Dimensions mismatch. Variance can not be calculated");
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static double [][] std(double[][] in,double[][] means){
        double [][] out=null;
        int meansRows=means.length;
        int meansCols=means[0].length;
        if (meansRows==1){
            out=new double[1][meansCols];
            for (int c=0;c<in[0].length;c++){

                for (int r=0;r<in.length;r++){
                    out[0][c]+=Math.pow(in[r][c]-means[0][c],2);
                    if (r==in.length-1){
                        out[0][c]=Math.sqrt(out[0][c]/in.length);
                    }
                }

            }
        }
        else if (meansCols==1){
            out=new double[meansRows][1];
            for (int r=0;r<in.length;r++){

                for (int c=0;c<in[0].length;c++){
                    out[r][0]+=Math.pow(in[r][c]-means[r][0],2);
                    if (c==in[0].length-1){
                        out[r][0]=Math.sqrt(out[r][0]/in[0].length) ;
                    }
                }

            }
        }
        else {
            throw new IllegalArgumentException("Dimensions mismatch. Variance can not be calculated");
        }

        return out;
    }


    public static double[][] transp(double [][] tensor){
        double [][] out=new double[tensor[0].length][tensor.length];
        for (int i=0;i<tensor.length;i++){
            for(int j=0;j<tensor[0].length;j++){
                out[j][i]=tensor[i][j];
            }
        }
        return out;
    }

    public static double[][] sum(double [][] data, Tensor.Dimension dim) {
        int rows=data.length;
        int cols=data[0].length;
        double sum[][]=null;
        switch (dim){
            case BYROWS :
                sum = new double[rows][1];
                for (int r=0;r<rows;r++){
                    sum[r][0]=sum(data[r]);
                }
                break;
            case BYCOLS:
                sum = new double[1][cols];
                for (int c=0;c<cols;c++){

                    for (int r=0;r<rows;r++){
                        sum[0][c]+=data[r][c];
                        if (r==rows-1){
                            sum[0][c]=sum[0][c];
                        }
                    }
                }
                break;
            case WHOLE:
                sum = new double[1][1];
                for (int i=0;i<rows;i++){

                    for (int j=0;j<cols;j++){
                        sum[0][0]+=data[i][j];

                    }
                }
                break;
        }


        return sum;
    }
    public static double sum(double [] vect) {
        double sum=0;
        for (int i=0;i<vect.length;i++){
                sum+=vect[i];

        }
        return sum;
    }

    public static double[][] invertValues(double[][] data){
        double [][] out=new double[data.length][data[0].length];
        for (int i=0;i< data.length;i++){
            for (int j=0;j<data[0].length;j++){
                out[i][j]=1/data[i][j];
            }
        }
        return out;
    }

    public static double[][] oneHot(double[][] data, int[] pos){
        double [][] out=new double[data.length][data[0].length];
        if (data.length!=pos.length){
            throw new IllegalArgumentException("Matrix size and position vector don't match");
        }
        for (int i=0;i< data.length;i++){
            for (int j=0; j<data[0].length;j++){
                if (j==pos[i]){
                    out[i][j]=1;
                }
                else
                    out[i][j]=0;

            }
        }
        return out;
    }

    public static double[][] randTensor(int rows, int cols){
        Random rand=new Random();
        double [][] data= new double [rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                data[i][j]=rand.nextGaussian();
            }
        }
        return data;
    }

    public static double[][] eye(int n){
        double [][] data= new double [n][n];
        for (int i=0;i<n;i++){
            for (int j=0;j<n;j++){
                if (i==j)
                    data[i][j]=1;
                else
                    data[i][j]=0;
            }
        }
        return data;
    }

    public static double[][] softMax(double [][] logits) {

        int rows=logits.length;
        int cols=logits[0].length;
        double out [][]= new double[rows][cols];
        int numberOfThreads = Runtime.getRuntime().availableProcessors();
        ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
        for (int i = 0; i < rows; i++) {
            final int row=i;

            executorService.execute(() -> {
                double max=0;
                double[] exp = new double[cols];
                double sumExp = 0.0;

                for (int j = 0; j < cols; j++) {
                    if (Math.abs(max)<Math.abs(logits[row][j]))
                        max=logits[row][j];
                }


                for (int j = 0; j < cols; j++) {
                    double correction=logits[row][j]-max;
                    exp[j] = Math.exp(correction);

                    sumExp += exp[j];
                }
                for (int j = 0; j < cols; j++) {
                    if (sumExp!=0)
                        out[row][j] = exp[j] / sumExp;
                    else {
                        throw new IllegalArgumentException("Softmax Overflow.");

                    }
                }
            });


        }
        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return out;
    }

    public static double crossEntropyLoss(double[][] probs,double[][] expData) {
        if (probs.length!=expData.length || probs[0].length!=expData[0].length){
            throw new IllegalArgumentException("Impossible to calculate Cross Entropy loss: Expected values don't match inputs.");
        }
        double totalLoss = 0.0;

        for (int i = 0; i < probs.length; i++) {
            for (int j = 0; j < probs[0].length; j++) {
                totalLoss += -expData[i][j] * Math.log(probs[i][j]);  // TODO: add +1e-5 for better numerical stability and fix derivation
                if (Double.isNaN(Math.log(probs[i][j]))){
                    throw new IllegalArgumentException("Overflow: log("+probs[i][j]+")");
                }
            }
        }
        return totalLoss/ probs.length;

    }

    public static double mse(double [][] dataa, double [][] datab){
        double out=0;
        if (dataa.length!=datab.length || dataa[0].length!=datab[0].length){
            throw new IllegalArgumentException("MSE not possible. Dimension doesn't match.");
        }
        for (int i=0;i< dataa.length;i++){
            for (int j=0;j< dataa[0].length;j++){
                out+=Math.pow(dataa[i][j]-datab[i][j],2);
            }
        }
        out=out/(dataa.length*dataa[0].length);
        return out;
    }

    public static double [][] ones(int rows,int cols){
        double [][] out=new double[rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                out[i][j]=1;
            }
        }
        return out;
    }

    public static double [][] batchnorm(double [][] data, double [][] means, double [][] vars,double epsilon){
        int rows=data.length;
        int cols=data[0].length;
        double [][] out=new double [rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                out[i][j]=(data[i][j]-means[0][j])/Math.pow(vars[0][j] + epsilon, 0.5);
            }
        }
        return out;
    }

    public static double [][] zeros(int rows,int cols){
        double [][] out=new double[rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                out[i][j]=0;
            }
        }
        return out;
    }

    public static double[][] hadamard(double [][] first, double [][] second){

        if (first.length!=second.length || first[0].length!=second[0].length){
            throw new IllegalArgumentException("Dimensions don't match; per item multiplication is not possible");
        }
        double [][] out=new double[first.length][first[0].length];
        for (int i=0;i<first.length;i++){
            for (int j=0;j<first[0].length;j++){
                out[i][j]=first[i][j]*second[i][j];
            }
        }
        return out;
    }

    public static double[][] muleach(double[][] data, double multiplier){
        double [][] out=new double[data.length][data[0].length];
        for (int i=0;i< data.length;i++){
            for (int j=0;j< data[0].length;j++){
                out[i][j]+=data[i][j]*multiplier;
            }
        }
        return out;
    }

    public static double[][] exp(double[][] data){
        double [][] out=new double[data.length][data[0].length];
        for (int i=0;i<data.length;i++){
            for (int j=0;j<data[0].length;j++){
                out[i][j]=Math.exp(data[i][j]);
            }
        }
        return out;
    }

    public static double[][] log(double[][] data){
        double [][] out=new double[data.length][data[0].length];
        for (int i=0;i<data.length;i++){
            for (int j=0;j<data[0].length;j++){
                out[i][j] = Math.log(data[i][j]);
            }
        }
        return out;
    }

    public static double[][] pow (double [][] data, double power){
        double [][] out=new double [data.length][data[0].length];
        for (int i=0;i<data.length;i++){
            for (int j=0;j<data[0].length;j++){
                out[i][j] = Math.pow(data[i][j],power);
            }
        }
        return out;
    }

    public static int[] generateMultinomialVector(int length, double[] probabilities, Random randomizer){
        double [] probs=probabilitiesNormalized(probabilities);


        int[] vector=new int[length];


        for (int i=0;i<vector.length;i++){
            double cumulative=0;
            double rand=randomizer.nextDouble();
            for (int j=0;j<probs.length;j++){
                cumulative+=probs[j];
                if (rand<cumulative){
                    vector[i]=j;
                    break;
                }
            }
        }
        return vector;

    }

    public static double[] probabilitiesNormalized(double[] probabilities){
        int sCount=probabilities.length;
        double[] probs = new double[sCount];
        double sum=0;
        for (int i=0;i<sCount;i++){
            sum+=probabilities[i];
        }
        for (int i=0;i<sCount;i++){
            probs[i]=probabilities[i]/sum;
        }
        return probs;
    }

    public static double [][] reshape (double [][] input, int newRows, int newCols){
        int rows=input.length;
        int cols=input[0].length;

        if ((newRows<=1) && (newCols<=1)){
            throw new IllegalArgumentException("Cannot reshape: both indexes are 0 or negative.");
        }
        else if ((newRows==-1) && (((rows*cols)%newCols==0))){
            newRows=(rows*cols)/newCols;
        }
        else if ((newCols==-1) && (((rows*cols)%newRows==0))){
            newCols=(rows*cols)/newRows;
        }
        double [][] out = new double [newRows][newCols];

        int k = 0;


        for (int i = 0; i < newRows; i++) {
            for (int j = 0; j < newCols; j++) {
                out[i][j] = input[k / input[0].length][k % input[0].length];
                k++;
            }
        }
        return out;
    }

    public static double [][][] reshape (double [][] input, int newRows, int newCols,int newDepth){
        int rows=input.length;
        int cols=input[0].length;

        if ((newRows<=1) && (newCols<=1 )&& (newDepth<=1)){
            throw new IllegalArgumentException("Cannot reshape: both indexes are 0 or negative.");
        }

        else if (newRows==-1){
            newRows=(rows*cols)/(newCols*newDepth);
        }
        else if (newCols==-1){
            newCols=(rows*cols)/(newRows*newDepth);
        }

        else if (newDepth==-1){
            newDepth=(rows*cols)/(newRows*newCols);
        }
        if (cols*rows!=(newRows*newCols*newDepth)){
            throw new IllegalArgumentException("Cannot reshape: (D:"+newDepth+"*R:"+newRows+"*C:"+newCols+") != initial R:"+rows+"*C:"+cols);
        }
        double [][][] out = new double [newDepth][newRows][newCols];

        int count = 0;


        for (int i = 0; i < newDepth; i++) {
            for (int j = 0; j < newRows; j++) {
                for (int k = 0; k < newCols; k++) {
                    out[k][i][j] = input[count / cols][count % cols];
                    //out[i][j][k] = input[count / (rows * cols)][(count / cols) % rows][count % cols];
                    count++;
                }
            }
        }
        return out;
    }



    public static double[][][] mapValues(double[][] rowIndexes, double[][] values) {
        int indexesRows = rowIndexes.length;
        int indexesCols = rowIndexes[0].length;

        int valuesRows = values.length;
        int valuesCols = values[0].length;


        double[][][] result = new double[indexesRows][indexesCols][valuesCols];

        for (int i = 0; i < indexesRows; i++) {
            for (int j = 0; j < indexesCols; j++) {
                int index = (int) rowIndexes[i][j];

                if (index >= 0 && index < values.length && j < values[index].length) {
                        //result[i][k+j] = values[index][k];
                   result[i][j]=values[index];
                }

            }
        }

        return result;
    }

    public static double[] calculateAverages(double[] input, int groups) {
        if (groups <= 0) {
            throw new IllegalArgumentException("Invalid value of n. It should be greater than 0.");
        }

        int groupSize = input.length / groups;
        double[] out = new double[groups];

        for (int i = 0; i < groups; i++) {
            int startIndex = i * groupSize;
            int endIndex = (i == groups - 1) ? input.length : (i + 1) * groupSize;
            double sum = Arrays.stream(Arrays.copyOfRange(input, startIndex, endIndex)).sum();
            out[i] = sum / (endIndex - startIndex);
        }

        return out;
    }


}
