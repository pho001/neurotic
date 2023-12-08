import org.ejml.simple.SimpleMatrix;

public class TestMath {
    private static double [][] testMatrixA=new double[][]{
        {-2,2,2},
        {3,-1,2},
        {3,5,7},
        {1,-1,9},
        {5,7,-2},
        {-4,-5,10},
        {0,15,20},
        {-20,20,7},
        {-3,6,9},
        {2,8,2},
        {6,-4,15},
        {21,-8,19},
        {2,5,0},
        {3,-5,8},
        {0,5,-2},
        {-20,20,7}
    };
    private static double [][] testMatrixB=new double [][]{
            {-2,2,2,8,7,6,5,5},
            {3,-1,2,5,4,2,4,1},
            {3,5,7,6,7,9,10,2}
    };


    public static void run(){
        print("Dot product",testDot());
        print("Addition",testAdd());
        print("Substraction",testSub());
        print("Mean",testMean());
        print("Variance",testVariance());
        print("Std",testStd());
        print("Sum",testSum());
        print("Normalization",testNorm());
        print("Substraction with broadcasting",testSubb());
        print("Addition with broadcasting",testAddb());
        print("Division with broadcasting",testDivb());
        print("Map Values test",testMapping());
    }

    private static boolean testDot(){
        SimpleMatrix test=new SimpleMatrix(MathHelper.dot(testMatrixA,testMatrixB));
        SimpleMatrix A=new SimpleMatrix(testMatrixA);
        SimpleMatrix B=new SimpleMatrix(testMatrixB);
        SimpleMatrix C=A.mult(B);
        boolean flag = C.isIdentical(test,0);
        return flag;
    }

    private static boolean testAdd(){
        SimpleMatrix test=new SimpleMatrix(MathHelper.add(testMatrixB,testMatrixB));
        SimpleMatrix A=new SimpleMatrix(testMatrixB);
        SimpleMatrix B=new SimpleMatrix(testMatrixB);
        SimpleMatrix C=A.plus(B);
        boolean flag = C.isIdentical(test,0);
        return flag;
    }
    private static boolean testSub(){
        SimpleMatrix test=new SimpleMatrix(MathHelper.sub(testMatrixB,testMatrixB));
        SimpleMatrix A=new SimpleMatrix(testMatrixB);
        SimpleMatrix B=new SimpleMatrix(testMatrixB);
        SimpleMatrix C=A.minus(B);
        boolean flag = C.isIdentical(test,0);
        return flag;
    }

    private static boolean testMean(){
        double [][] testMatrix=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] expByCol=new double[][]{
                {1.75,2.75,4.5,6.25}
        };
        double [][] expByRow=new double[][]{
                {2.5},{2.25},{5.25},{5.25}
        };
        double [][] expByAll=new double[][]{
                {3.8125}
        };
        boolean flag=true;
        double [][] meanByCol=MathHelper.mean(testMatrix, Tensor.Dimension.BYCOLS);
        double [][] meanByRow=MathHelper.mean(testMatrix, Tensor.Dimension.BYROWS);
        double [][] meanByWhole=MathHelper.mean(testMatrix, Tensor.Dimension.WHOLE);
        for (int i=0;i<expByRow.length;i++){
            for (int j=0;j<expByRow[0].length;j++){
                if (meanByRow[i][j]!=expByRow[i][j]){
                    flag=false;
                }
            }
        }
        for (int i=0;i<expByCol.length;i++){
            for (int j=0;j<expByCol[0].length;j++){
                if (meanByCol[i][j]!=expByCol[i][j]){
                    flag=false;
                }
            }
        }
        for (int i=0;i<expByAll.length;i++){
            for (int j=0;j<expByAll[0].length;j++){
                if (meanByWhole[i][j]!=expByAll[i][j]){
                    flag=false;
                }
            }
        }
        return flag;

    }

    private static boolean testVariance(){
        double [][] testMatrix=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] expCols=new double[][]{
                {6.2500,8.2500,8.3333,1.5833}
        };
        double [][] expRows=new double[][]{
                {17},{6.25},{2.9167},{2.9167}
        };

        boolean flag=true;
        double[][] mean;
        double[][] calc;
        mean=MathHelper.mean(testMatrix, Tensor.Dimension.BYCOLS);
        calc=MathHelper.variance(testMatrix, mean);
        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expCols[0].length;j++){
                if (Math.abs(calc[i][j]-expCols[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        mean=MathHelper.mean(testMatrix, Tensor.Dimension.BYROWS);
        calc=MathHelper.variance(testMatrix, mean);
        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expRows[0].length;j++){
                if (Math.abs(calc[i][j]-expRows[i][j])>=1e-4){
                    flag=false;
                }
            }
        }


        return flag;

    }


    private static boolean testStd(){
        double [][] testMatrix=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] expCols=new double[][]{
                {2.1651,2.4875,2.5,1.0897}
        };
        double [][] expRows=new double[][]{
                {3.5707},{2.1651},{1.4790},{1.4790}
        };

        boolean flag=true;
        double[][] mean;
        double[][] calc;
        mean=MathHelper.mean(testMatrix, Tensor.Dimension.BYCOLS);
        calc=MathHelper.std(testMatrix,mean);

        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expCols[0].length;j++){
                if (Math.abs(calc[i][j]-expCols[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        mean=MathHelper.mean(testMatrix, Tensor.Dimension.BYROWS);
        calc=MathHelper.std(testMatrix, mean);
        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expRows[0].length;j++){
                if (Math.abs(calc[i][j]-expRows[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        return flag;

    }

    private static boolean testSum() {
        double[][] testMatrix = new double[][]{
                {-2, 2, 2, 8},
                {3, -1, 2, 5},
                {3, 5, 7, 6},
                {3, 5, 7, 6},
        };
        double exp=61;
        double calc=MathHelper.sum(testMatrix, Tensor.Dimension.WHOLE)[0][0];
        boolean flag=true;
        if (Math.abs(exp-calc)>=1e-4){
            flag=false;
        }

        double exp1[][]= new double[][]{
                {10},
                {9},
                {21},
                {21}
        };
        double calc1[][]=MathHelper.sum(testMatrix, Tensor.Dimension.BYROWS);

        for (int i=0;i< calc1.length;i++){
            for (int j=0;j<exp1[0].length;j++){
                if (Math.abs(calc1[i][j]-exp1[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        exp1= new double[][]{
                {7,11,18,25}
        };
        calc1=MathHelper.sum(testMatrix, Tensor.Dimension.BYCOLS);
        for (int i=0;i< calc1.length;i++){
            for (int j=0;j<exp1[0].length;j++){
                if (Math.abs(calc1[i][j]-exp1[i][j])>=1e-4){
                    flag=false;
                }
            }
        }
        testMatrix = new double[][]{
                {-2, 2, 2, 8}

        };
        calc=MathHelper.sum(testMatrix[0]);
        exp=10;
        if (Math.abs(exp-calc)>=1e-4){
            flag=false;
        }
        return flag;
    }


    private static boolean testNorm(){
        double [][] testMatrix=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] expCols=new double[][]{
                {-1.7321,  -0.3015,  -1.0000,   1.6059},
                {0.5774,  -1.5076,  -1.0000, -1.1471},
                {0.5774,   0.9045,   1.0000,  -0.2294},
                {0.5774,   0.9045,   1.0000,  -0.2294}
        };
        double [][] expRows=new double[][]{
                {-1.2603,  -0.1400,  -0.1400,   1.5403},
                {0.3464, -1.5011,  -0.1155,   1.2702},
                {-1.5213,  -0.1690,   1.1832,   0.5071},
                {-1.5213,  -0.1690,   1.1832,   0.5071}
        };

        boolean flag=true;
        double[][] mean;
        double[][] calc;

        calc=MathHelper.normalize(testMatrix,Tensor.Dimension.BYCOLS);

        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expCols[0].length;j++){
                if (Math.abs(calc[i][j]-expCols[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        calc=MathHelper.normalize(testMatrix,Tensor.Dimension.BYROWS);
        for (int i=0;i<calc.length;i++){
            for (int j=0;j<expRows[0].length;j++){
                if (Math.abs(calc[i][j]-expRows[i][j])>=1e-4){
                    flag=false;
                }
            }
        }

        return flag;

    }

    private static void print(String what,boolean f){
        String color="";
        if (f){
            color="\u001B[32m";
        }
        else{
            color="\u001B[31m";
        }
        System.out.println(what+" test result: "+color+f+"\u001B[0m");
    }

    private static boolean testAddb(){
        double [][] testA=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] testColBroadcast=new double [][]{
                {-2},
                {3},
                {3},
                {3},
        };
        double [][] testRowBroadcast=new double [][]{
                {-2,2,2,8}
        };
        double [][] calcColBroadcast=new double [][]{
                {-4,   0,    0,    6},
                {6,    2,    5,    8},
                {6,    8,   10,    9},
                {6,    8,   10,    9}
        };
        double [][] calcRowBroadcast=new double [][]{
                {-4,   4,    4,   16},
                {1,    1,    4,   13},
                {1,    7,    9,   14},
                {1,    7,    9,   14}

        };

        double [][] testRowsBroadcasted=MathHelper.addb(testA,testRowBroadcast);
        double [][] testColsBroadcasted=MathHelper.addb(testA,testColBroadcast);

        boolean flag=true;
        for (int i=0;i<testA.length;i++){
            for(int j=0;j<testA[0].length;j++){
                if (testRowsBroadcasted[i][j]!=calcRowBroadcast[i][j]){
                    flag=false;
                }
                if (testColsBroadcasted[i][j]!=calcColBroadcast[i][j]){
                    flag=false;
                }
            }
        }

        return flag;
    }

    private static boolean testSubb(){
        double [][] testA=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] testColBroadcast=new double [][]{
                {-2},
                {3},
                {3},
                {3},
        };
        double [][] testRowBroadcast=new double [][]{
                {-2,2,2,8}
        };
        double [][] calcColBroadcast=new double [][]{
                {0,    4,    4,   10},
                {0,   -4,   -1,   2},
                {0,    2,    4,    3},
                {0,    2,    4,   3},
        };
        double [][] calcRowBroadcast=new double [][]{
                {0,   0,   0,   0},
                {5,  -3,   0,  -3},
                {5,   3,   5,  -2},
                {5,   3,   5,  -2},

        };

        double [][] testRowsBroadcasted=MathHelper.subb(testA,testRowBroadcast);
        double [][] testColsBroadcasted=MathHelper.subb(testA,testColBroadcast);

        boolean flag=true;
        for (int i=0;i<testA.length;i++){
            for(int j=0;j<testA[0].length;j++){
                if (testRowsBroadcasted[i][j]!=calcRowBroadcast[i][j]){
                    flag=false;
                }
                if (testColsBroadcasted[i][j]!=calcColBroadcast[i][j]){
                    flag=false;
                }
            }
        }

        return flag;
    }


    private static boolean testDivb(){
        double [][] testA=new double [][]{
                {-2,2,2,8},
                {3,-1,2,5},
                {3,5,7,6},
                {3,5,7,6},
        };
        double [][] testColBroadcast=new double [][]{
                {-2},
                {3},
                {3},
                {3},
        };
        double [][] testRowBroadcast=new double [][]{
                {-2,2,2,8}
        };
        double [][] calcColBroadcast=new double [][]{
                {1.0000,  -1.0000,  -1.0000,  -4.0000},
                {1.0000,  -0.3333,   0.6667,   1.6667},
                {1.0000,   1.6667,   2.3333,   2.0000},
                {1.0000,   1.6667,   2.3333,   2.0000},
        };
        double [][] calcRowBroadcast=new double [][]{
                {1.0000,   1.0000,  1.0000,   1.0000},
                {-1.5000,  -0.5000,   1.0000,   0.6250},
                {-1.5000,   2.5000,   3.5000,   0.7500},
                {-1.5000,   2.5000,   3.5000,   0.7500}

        };

        double [][] testRowsBroadcasted=MathHelper.divb(testA,testRowBroadcast);
        double [][] testColsBroadcasted=MathHelper.divb(testA,testColBroadcast);

        boolean flag=true;
        for (int i=0;i<testA.length;i++){
            for(int j=0;j<testA[0].length;j++){
                if (Math.abs(testRowsBroadcasted[i][j]-calcRowBroadcast[i][j])>1e-4){
                    flag=false;
                }
                if (Math.abs(testColsBroadcasted[i][j]-calcColBroadcast[i][j])>1e-4){
                    flag=false;
                }
            }
        }

        return flag;
    }

    private static boolean testMapping() {
        boolean flag = true;
        double[][] values = new double[][]{
                {0.9746, 0.9883, 0.1969, 0.2840, 0.0899},
                {0.3269, 0.5826, 0.8740, 0.6230, 0.8120},
                {0.2351, 0.3033, 0.4513, 0.8638, 0.4302},
                {0.3233, 0.2734, 0.7830, 0.9366, 0.4599}
        };
        double[][] indexes = new double[][]{
                {1, 2, 3},
                {2, 3, 2}
        };

        double[][][] resultExp = new double[][][]{
                {
                        {0.3269, 0.5826, 0.8740, 0.6230, 0.8120},
                        {0.2351, 0.3033, 0.4513, 0.8638, 0.4302},
                        {0.3233, 0.2734, 0.7830, 0.9366, 0.4599}

                },
                {

                        {0.2351, 0.3033, 0.4513, 0.8638, 0.4302},
                        {0.3233, 0.2734, 0.7830, 0.9366, 0.4599},
                        {0.2351, 0.3033, 0.4513, 0.8638, 0.4302}

                }
        };
        double[][][] resultCalc = MathHelper.mapValues(indexes, values);

        for (int i = 0; i < resultExp.length; i++) {
            for (int j = 0; j < resultExp[0].length; j++) {
                for (int k = 0; k < resultExp[0][0].length; k++) {
                    if (Math.abs(resultExp[i][j][k] - resultCalc[i][j][k]) > 1e-4) {
                        flag = false;
                    }
                }
            }
        }

        return flag;
    }
}
