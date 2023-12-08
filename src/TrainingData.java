public class TrainingData {
    public double[][] trainingData;
    public double[][] expectedoutputs;
    double[][] control_data;

    public TrainingData(){
        this.trainingData=new double[][]{
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
        this.expectedoutputs=new double[][]{
                {2},
                {4},
                {15},
                {9},
                {10},
                {1},
                {35},
                {7},
                {12},
                {12},
                {17},
                {32},
                {7},
                {6},
                {3},
                {7}
        };

        this.control_data=new double[][]{
                        {5,5,-2},
                        {-4,-3,10},
                        {-5,15,20},
                        {-2,2,7}
        };

    }
}

