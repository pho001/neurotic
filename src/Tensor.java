import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.algo.DenseDoubleAlgebra;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;
import cern.jet.math.Functions;



import java.util.*;




public class Tensor {

        public double[][] data;
        public int[] shape;
        public int rows;
        public int cols;
        private Set<Tensor> _prev;

        private Runnable localgradients;
        public double[][] gradients;
        public String label;

        double gamma;
        double beta;


        public enum Join{
            LEFT,
            RIGHT,
            ABOVE,
            UNDER
        }

        public enum Rndtype{
            INT,
            DOUBLE

        }

        public enum Norm{
            BYVAR,
            BYSTD

        }
        public enum Dimension{
            BYROWS,
            BYCOLS,
            WHOLE
        }




        public Tensor(int rows,int cols, Set<Tensor> _prev,String label) {
            this.data=new double[rows][cols];
            this.shape=new int[] {rows,cols};
            this.rows=rows;
            this.cols=cols;
            this._prev=new HashSet<>(_prev);
            this.localgradients=()->{};
            this.gradients=new double[rows][cols];
            this.label=label;

        }

        public Tensor (double [][] data, Set<Tensor> _prev, String label){
            this.data=data;
            this.rows=data.length;
            this.cols=data[0].length;
            this.shape=new int[] {data.length,data[0].length};
            this._prev=new HashSet<>(_prev);
            this.localgradients=()->{};
            this.gradients=new double[rows][cols];
            this.label=label;
        }



        public Tensor add (Tensor secondMatrix){


            Tensor out = new Tensor(MathHelper.add(this.data,secondMatrix.data), Set.of(this,secondMatrix),this.label+"+"+secondMatrix.label);

            out.localgradients=()->{
                for (int i=0;i<this.rows;i++){
                    for (int j=0;j<cols;j++){
                        this.gradients[i][j]=+1*out.gradients[i][j];
                        secondMatrix.gradients[i][j]+=1*out.gradients[i][j];
                    }
                }

            };


            return out;
        }

        //addition with broadcasting
        public Tensor addb (Tensor secondMatrix){
            Tensor out = new Tensor(MathHelper.addb(this.data,secondMatrix.data), Set.of(this,secondMatrix),this.label+"+"+secondMatrix.label);
            //gradients for broadcasting around second dimension must be added
            out.localgradients=()->{

                for (int i=0;i<this.rows;i++){

                    for (int j=0;j<this.cols;j++){
                        this.gradients[i][j]+=out.gradients[i][j];
                        if (secondMatrix.rows==1){
                            secondMatrix.gradients[0][j]+=out.gradients[i][j];
                        }
                        else if(secondMatrix.cols==1){
                            secondMatrix.gradients[i][0]+=out.gradients[i][j];
                        }
                        else {
                            throw new IllegalArgumentException("Unable to broadcast. Denominator can be either row or column vector.");
                        }

                    }

                }


            };
            return out;
        }

        public Tensor sub (Tensor secondMatrix){
            Tensor out=new Tensor(this.rows, this.cols, Set.of(this,secondMatrix),this.label+"-"+secondMatrix.label);

            out.data=MathHelper.sub(this.data, secondMatrix.data);

            out.localgradients=()->{

                for (int i=0;i<this.rows;i++){
                    for (int j=0;j<cols;j++){
                        this.gradients[i][j]+=-1*out.gradients[i][j];
                        secondMatrix.gradients[i][j]+=-1*out.gradients[i][j];
                    }

                }

            };

            return out;
        }

        public Tensor subb (Tensor secondMatrix){
            Tensor out=new Tensor(this.rows, this.cols, Set.of(this,secondMatrix),this.label+"-"+secondMatrix.label);
            out.data=MathHelper.subb(this.data, secondMatrix.data);
            //gradients for broadcasting around second dimension must be added
            out.localgradients=()->{
                for (int i=0;i<this.rows;i++){
                    for (int j=0;j<this.cols;j++){
                        //this.gradients[i][j]+=-1*out.gradients[i][j];
                        this.gradients[i][j]+=out.gradients[i][j];
                        if (secondMatrix.rows==1){
                            secondMatrix.gradients[0][j]+=-out.gradients[i][j];
                        }
                        else if(secondMatrix.cols==1){
                            secondMatrix.gradients[i][0]+=-out.gradients[i][j];
                        }
                        else {
                            throw new IllegalArgumentException("Unable to broadcast. Denominator can be either row or column vector.");
                        }

                    }
                }

            };
            return out;
        }




        public Tensor dot (Tensor secondMatrix) {
            Tensor out = new Tensor(this.rows, secondMatrix.cols, Set.of(this, secondMatrix), "(" +this.label + "×" + secondMatrix.label+ ")" );
            //out.data=MathHelper.dot(this.data,secondMatrix.data);
            DoubleMatrix2D matrixA = new DenseDoubleMatrix2D(this.data);
            DoubleMatrix2D matrixB = new DenseDoubleMatrix2D(secondMatrix.data);

            DenseDoubleAlgebra algebra = new DenseDoubleAlgebra();
            DoubleMatrix2D result = algebra.mult(matrixA, matrixB);
            out.data=result.toArray();


            out.localgradients=()->{
                //secondMatrix.gradients=MathHelper.add(secondMatrix.gradients,MathHelper.dot(MathHelper.transp(this.data), out.gradients));
                //this.gradients=MathHelper.add(this.gradients,MathHelper.dot(out.gradients, MathHelper.transp(secondMatrix.data)));
                DoubleMatrix2D outGrads= new DenseDoubleMatrix2D(out.gradients);
                secondMatrix.gradients=algebra.mult(algebra.transpose(matrixA), outGrads).assign(new DenseDoubleMatrix2D(secondMatrix.gradients),(a, b)->a+b).toArray();
                this.gradients=algebra.mult(outGrads,algebra.transpose(matrixB)).assign(new DenseDoubleMatrix2D(this.gradients),(a, b)->a+b).toArray();


                /*
                final double [][] secondMatrixGrad=new double[this.cols][out.cols];
                final double [][] firstMatrixGrad=new double[out.rows][secondMatrix.rows];

                //let's use threads top make dot products faster
                int numberOfThreads = Runtime.getRuntime().availableProcessors();
                ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
                //ExecutorService executorService = Executors.newVirtualThreadPerTaskExecutor();

                //second matrix gradients
                for (int i = 0; i < secondMatrixGrad.length; i++) {
                    final int row=i;
                    executorService.execute(() ->{
                        for (int j = 0; j < secondMatrixGrad[0].length; j++) {
                            secondMatrixGrad[row][j] = 0;
                            for (int k = 0; k < this.rows; k++) {
                                secondMatrixGrad[row][j] += this.data[k][row] * out.gradients[k][j];
                            }
                            secondMatrix.gradients[row][j]+=secondMatrixGrad[row][j];
                        }

                    });
                }
                executorService.shutdown();

                try {
                    executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                executorService = Executors.newFixedThreadPool(numberOfThreads);

                //first matrix gradients
                for (int i = 0; i < this.rows; i++) {
                    final int row=i;
                    executorService.execute(() ->{
                        for (int j = 0; j < this.cols; j++) {
                            firstMatrixGrad[row][j] = 0;
                            for (int k = 0; k < out.cols; k++) {
                                firstMatrixGrad[row][j] += out.gradients[row][k] * secondMatrix.data[j][k];
                            }
                            this.gradients[row][j]+=firstMatrixGrad[row][j];
                        }

                    });
                }
                executorService.shutdown();

                try {
                    executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                */





            };

            return out;
        }


        public Tensor log (){
            Tensor out=new Tensor(MathHelper.log(this.data), Set.of(this), "Log(" + this.label+")");
            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=1/this.data[i][j]*out.gradients[i][j];
                    }
                }
            };

            return out;
        }

        public Tensor batchNorm(Tensor Beta, Tensor Gamma, double [][] mean, double[][] variance, double epsilon) {

            //double[][] mean=MathHelper.mean(this.data,Dimension.BYCOLS);
            //double[][] var=MathHelper.variance(this.data,mean);
            double [][] normalized=new double [this.rows][this.cols];
            double N=this.data.length;

            //AtomicInteger numberOfThreads = new AtomicInteger(Runtime.getRuntime().availableProcessors());
            //AtomicReference<ExecutorService> executorService = new AtomicReference<>(Executors.newFixedThreadPool(numberOfThreads.get()));

            Tensor out=new Tensor(this.rows,this.cols,Set.of(this,Beta,Gamma),"batchnorm("+this.label+")");
            for (int i=0;i<out.data.length;i++){
                final int row=i;
               // executorService.get().execute(() ->{
                for (int j=0;j<out.data[0].length;j++){
                    normalized[row][j]=(this.data[row][j]-mean[0][j])/(Math.pow(variance[0][j]+epsilon,0.5));
                    out.data[row][j]=Gamma.data[0][j]*normalized[row][j]+Beta.data[0][j];
                }
                //});
            }

            //executorService.get().shutdown();
            /*
            try {
                executorService.get().awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            */
            out.localgradients=()->{
                double [][] mul=MathHelper.hadamard(out.gradients,normalized);
                double [][] sum=MathHelper.sum(mul, Dimension.BYCOLS);
                Gamma.gradients=sum;
                double [][] sum_grad=MathHelper.sum(out.gradients, Dimension.BYCOLS);
                Beta.gradients=sum_grad;

                //numberOfThreads.set(Runtime.getRuntime().availableProcessors());
                //executorService.set(Executors.newFixedThreadPool(numberOfThreads.get()));
                for (int i=0;i<out.rows;i++){
                    final int row=i;
                    //executorService.get().execute(() ->{
                    for (int j=0;j<out.cols;j++){
                        double outer_term=(Gamma.data[0][j]*Math.pow((variance[0][j]+epsilon),-0.5))/N;
                        double first_inner_term=N*out.gradients[row][j];
                        double second_inner_term=-sum_grad[0][j];
                        double third_inner_term=-N/(N-1)*normalized[row][j]*sum[0][j];
                        this.gradients[row][j]=outer_term*(first_inner_term+second_inner_term+third_inner_term);
                    }
                    //});
                }
                //executorService.get().shutdown();
                /*
                try {
                    executorService.get().awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

                 */

            };

            return out;
        }

        public Tensor std(Tensor mean){
            Tensor out=this.stdinternal(mean);
            return out;
        }

        public Tensor std(Dimension dim){
            Tensor mean=this.mean(dim);
            Tensor out=new Tensor(MathHelper.std(this.data,mean.data), Set.of(this,mean),"Std(" + this.label+")");
            out.localgradients=()->{
                //by cols - mean and std are 1*cols vector
                int rows=this.rows;
                int cols=this.cols;
                if (mean.data.length==1){
                    for (int c=0;c<cols;c++){
                        double sum=0;
                        for (int r=0;r<rows;r++){
                            sum+=this.data[r][c]-mean.data[0][c];
                        }
                        for (int r=0;r<rows;r++){
                            // wrt mean
                            mean.gradients[0][c]=(-sum/(rows*out.data[0][c]))*out.gradients[0][c];
                            //wrt x
                            this.gradients[r][c]=(sum/(rows*out.data[0][c]))*out.gradients[r][c];
                        }
                    }

                }
                //by rows - mean and std are rows*1 vector
                else if (mean.data[0].length==1){

                    for (int r=0;r<rows;r++){
                        double sum=0;
                        for (int c=0;c<cols;c++){
                            sum+=this.data[r][c]-mean.data[c][0];
                        }

                        for (int c=0;c<cols;c++){
                            // wrt mean
                            mean.gradients[r][0]=(-sum/(rows*out.data[r][0]))*out.gradients[r][0];
                            //wrt x
                            this.gradients[r][c]=(sum/(rows*out.data[r][0]))*out.gradients[r][0];
                        }
                    }

                }
                else{
                    throw new IllegalArgumentException("Incorrect mean dimension.");
                }

            };

            return out;
        }

        private Tensor stdinternal(Tensor mean){
            Tensor out=new Tensor(MathHelper.std(this.data,mean.data), Set.of(this,mean),"Std(" + this.label+")");

            out.localgradients=()->{
                //by cols - mean and std are 1*cols vector
                int rows=this.rows;
                int cols=this.cols;
                if (mean.data.length==1){
                    for (int c=0;c<cols;c++){
                        double sum=0;
                        for (int r=0;r<rows;r++){
                            sum+=this.data[r][c]-mean.data[0][c];
                        }
                        for (int r=0;r<rows;r++){
                            // derivative of std wrt mean
                            mean.gradients[0][c]+=(-sum/(rows*out.data[0][c]))*out.gradients[0][c];
                            //wrt x
                            this.gradients[r][c]+=(sum/(rows*out.data[0][c]))*out.gradients[0][c];
                        }
                    }

                }
                //by rows - mean and std are rows*1 vector
                else if (mean.data[0].length==1){

                    for (int r=0;r<rows;r++){
                        double sum=0;
                        for (int c=0;c<cols;c++){
                            sum+=this.data[r][c]-mean.data[c][0];
                        }

                        for (int c=0;c<cols;c++){
                            // derivative of std wrt mean
                            mean.gradients[r][0]+=(-sum/(rows*out.data[r][0]))*out.gradients[r][0];
                            //wrt x
                            this.gradients[r][c]+=(sum/(rows*out.data[r][0]))*out.gradients[r][0];
                        }
                    }

                }
                else{
                    throw new IllegalArgumentException("Incorrect mean dimension.");
                }

            };
            return out;
        }


        public Tensor exp (){
            Tensor out=new Tensor(MathHelper.exp(this.data), Set.of(this), "Exp(" + this.label+")");

            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=Math.exp(this.data[i][j])*out.gradients[i][j];
                    }
                }
            };

            return out;
        }

        public Tensor pow(double power){
            Tensor out=new Tensor(this.rows,this.cols, Set.of(this), "pow(" + this.label+","+power+")");
            out.data=MathHelper.pow(this.data, power);
            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=power*Math.pow(this.data[i][j],power-1)*out.gradients[i][j];
                    }
                }
            };
            return out;
        }



        public Tensor mean(Dimension dim){
            Tensor out=new Tensor(MathHelper.mean(this.data,dim), Set.of(this), "Mean(" + this.label+")");


            out.localgradients=()->{
                double [][] help=null;
                double n=0;
                switch (dim){
                    case BYCOLS :
                        n=(double)1/this.rows;
                        help=MathHelper.muleach(out.gradients,n);
                        for (int i=0;i<this.rows;i++){
                            for (int j=0;j<this.cols;j++){
                                this.gradients[i][j]+=help[0][j];
                            }
                        }
                        break;
                    case BYROWS:
                        n=(double)1/this.cols;
                        help=MathHelper.muleach(out.gradients,n);
                        for (int i=0;i<this.rows;i++){
                            for (int j=0;j<this.cols;j++){
                                this.gradients[i][j]+=help[i][0];
                            }
                        }
                        break;
                    case WHOLE:
                        break;
                }

            };

            return out;
        }
        public Tensor variance(Tensor mean){
            Tensor out=new Tensor(MathHelper.variance(this.data,mean.data), Set.of(this,mean), "Variance(" + this.label+")");
            out.localgradients=()->{

                int rows=this.rows;
                int cols=this.cols;
                //by cols
                if (mean.data.length==1){
                    for (int c=0;c<cols;c++){
                        double sum=0;
                        for (int r=0;r<rows;r++){
                            sum+=this.data[r][c]-mean.data[0][c];
                        }
                        for (int r=0;r<rows;r++){
                            // wrt mean
                            //mean.gradients[r][c]+=(-2/rows)*sum*out.gradients[r][c];
                            //wrt x
                            this.gradients[r][c]+=(2/(double)(rows-1))*(this.data[r][c]-mean.data[0][c])*out.gradients[0][c];

                            mean.gradients[0][c]+=0;

                        }
                    }

                }
                //by rows
                else if (mean.data[0].length==1){

                    for (int r=0;r<rows;r++){
                        double sum=0;
                        for (int c=0;c<cols;c++){
                            sum+=this.data[r][c]-mean.data[c][0];
                        }

                        for (int c=0;c<cols;c++){
                            //wrt mean
                            //mean.gradients[r][c]+=(-2/cols)*sum*out.gradients[r][c];
                            //wrt x
                            this.gradients[r][c]+=this.gradients[r][c]+=(2/(double)(rows-1))*(this.data[r][c]-mean.data[r][0])*out.gradients[r][0];
                            mean.gradients[r][0]+=0;

                        }
                    }

                }
                else{
                    throw new IllegalArgumentException("Incorrect mean dimension.");
                }
            };
            return out;
        }




        public Tensor get(int rowIndex, int colIndex) {


            if (rowIndex <-1 || colIndex<-1) {
                throw new IllegalArgumentException("Selectors are empty.");
            }

            try {
                if (rowIndex>-1 && colIndex>-1)
                {
                    double [][] pom=new double[1][1];
                    pom[0][0]=this.data[rowIndex][colIndex];
                    Tensor out = new Tensor(pom,new HashSet<>(),"("+rowIndex+","+colIndex+")");
                    return out;
                }
                else if (colIndex==-1) {
                    if (rowIndex < 0 || rowIndex >= this.data.length) {
                        throw new IllegalArgumentException("Incorrect row index");
                    }


                    double[][] pom = new double[1][this.data[rowIndex].length];
                    pom[0]=this.data[rowIndex];
                    Tensor out=new Tensor(pom,new HashSet<>(),"("+rowIndex+","+colIndex+")");
                    return out;
                }

                else if (rowIndex==-1) {

                    if (this.data.length == 0 || colIndex < 0 || colIndex >= this.data[0].length) {
                        throw new IllegalArgumentException("Incorrect column index.");
                    }

                    double[][] selectedColumn = new double[this.data.length][1];
                    for (int i = 0; i < this.data.length; i++) {
                        selectedColumn[i][0] = this.data[i][colIndex];
                    }

                    Tensor out=new Tensor(selectedColumn,new HashSet<>(),"("+rowIndex+","+colIndex+")");
                    return out;
                }
                else if (colIndex==-1 && rowIndex==-1) {
                    Tensor out=this;
                    return out;
                }


                throw new IllegalArgumentException("Wrong selector.");

            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Wrong selector.");
            }


        }

        //per element division with broadcasting
        public Tensor div (Tensor denominator){
            Tensor out=new Tensor(MathHelper.divb(this.data,denominator.data),Set.of(this,denominator),this.label+"/"+denominator.label);
            //this is just for denominator as row vector. needs to be fixed
            out.localgradients=()->{

                    for (int i = 0; i < out.rows; i++) {
                        for (int j = 0; j < out.cols; j++) {
                            if (denominator.rows==1){
                                this.gradients[i][j] += 1 / denominator.data[0][j] * out.gradients[i][j];
                                //denominator.gradients[0][j] += this.data[i][j] * out.gradients[i][j];
                                denominator.gradients[0][j] += (-this.data[i][j]/Math.pow(denominator.data[0][j],2)) * out.gradients[i][j];
                            }
                            else if(denominator.cols==1){
                                this.gradients[i][j] += 1 / denominator.data[i][0] * out.gradients[i][j];
                                //denominator.gradients[i][0] += this.data[i][j] * out.gradients[i][j];
                                denominator.gradients[i][0] += (-this.data[i][j]/Math.pow(denominator.data[i][0],2)) * out.gradients[i][j];
                            }
                            else {
                                throw new IllegalArgumentException("Unable to broadcast. Denominator can be either row or column vector.");
                            }
                        }

                    }

            };
            return out;
        }

        public Tensor join(Tensor B,Join join) {
            Tensor out;
            if (join==Join.LEFT || join==Join.RIGHT){

                if (this.rows!=B.rows){
                    throw new IllegalArgumentException("Rows don't match");

                }
                out= new Tensor(this.rows,this.cols+B.cols,Set.of(this,B)," "+join+" ("+this.label+"+"+B.label+")");
            }
            else if (join==Join.ABOVE || join==Join.UNDER){
                if (this.cols!=B.cols){
                    throw new IllegalArgumentException("Cols don't match");
                }
                out= new Tensor(this.rows+B.rows,this.cols,new HashSet<>(), " "+join+" ("+this.label+"+"+B.label+")");
            }
            else{
                throw new IllegalArgumentException("Join is not specified");
            }

            switch (join) {
                case LEFT:
                    for (int i = 0; i < out.rows; i++) {
                        for (int j = 0; j < B.cols; j++) {
                            out.data[i][j] = B.data[i][j];
                        }
                        for (int j = 0; j < this.cols; j++) {
                            out.data[i][B.cols + j] = this.data[i][j];
                        }
                    }
                    break;

                case RIGHT:
                    for (int i = 0; i < out.rows; i++) {
                        for (int j = 0; j < this.cols; j++) {
                            out.data[i][j] = this.data[i][j];
                        }
                        for (int j = 0; j < B.cols; j++) {
                            out.data[i][this.cols + j] = B.data[i][j];
                        }
                    }
                    break;
                case ABOVE:
                    for (int i = 0; i < B.rows; i++) {
                        out.data[i] = B.data[i];

                    }
                    for (int i = 0; i < this.rows; i++) {
                        out.data[B.rows + i] = this.data[i];

                    }
                    break;
                case UNDER:
                    for (int i = 0; i < this.rows; i++) {

                        out.data[i] = this.data[i];

                    }
                    for (int i = 0; i < B.rows; i++) {
                        out.data[this.rows + i] = B.data[i];

                    }
                    break;


            }
            out.localgradients=()->{

                for (int i=0;i<this.rows;i++){
                    for (int j=0;j<this.cols;j++){
                        switch (join) {
                            case LEFT:
                                if (j+B.cols<out.cols) {
                                    this.gradients[i][j] = 1 * out.gradients[i][j + B.cols];
                                }
                                B.gradients[i][j] = 1 * out.gradients[i][j];
                                break;
                            case RIGHT:
                                this.gradients[i][j] = 1 * out.gradients[i][j];
                                if (j+this.cols<out.cols) {
                                    B.gradients[i][j] = 1 * out.gradients[i][j + this.cols];
                                }
                                break;
                            case ABOVE:
                                if (i+B.rows<out.rows) {
                                    this.gradients[i][j] = 1 * out.gradients[i + B.rows][j];
                                }
                                B.gradients[i][j] = 1 * out.gradients[i][j];
                                break;
                            case UNDER:
                                this.gradients[i][j] = 1 * out.gradients[i][j];
                                if (j+this.rows<out.rows) {
                                    B.gradients[i][j] = 1 * out.gradients[i + this.rows][j];
                                }
                                break;
                        }
                    }
                }
            };
            return out;
        }

        public Tensor  reshape(int newRows, int newCols) {
            if ((this.rows * this.cols != newRows * newCols) && ((newRows>1) && (newCols>1))) {
                throw new IllegalArgumentException("Amount of elements in new shape doesn't match amount of elements in od shape.");
            }


            Tensor out=new Tensor (MathHelper.reshape(this.data,newRows,newCols),Set.of(this),"Reshape of ("+this.label+")");

            out.localgradients=()->{
                int k = 0;
                for (int i = 0; i < this.rows; i++) {
                    for (int j = 0; j < this.cols; j++) {
                        this.gradients[i][j] += out.gradients[k / out.cols][k % out.cols];
                        k++;
                    }
                }
            };

            return out;
        }

        public Tensor muleach(double multiplier){
            Tensor out=new Tensor(MathHelper.muleach(this.data,multiplier),new HashSet<>(),".*"+multiplier);
            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    double sum=MathHelper.sum(out.data[i]);
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=multiplier*out.gradients[i][j];
                    }
                }
            };
            return out;
        }


    //per element multiplication with broadcasting
    public Tensor mulb (Tensor multiplicator){
        Tensor out=new Tensor(MathHelper.mulb(this.data,multiplicator.data),Set.of(this,multiplicator),this.label+"*"+multiplicator.label);
        //this is just for multiplicator as row vector. needs to be fixed
        out.localgradients=()->{
                for (int i = 0; i < out.rows; i++) {
                    for (int j = 0; j < out.cols; j++) {
                        if (multiplicator.rows==1){
                            this.gradients[i][j] += multiplicator.data[0][j] * out.gradients[i][j];
                            multiplicator.gradients[0][j] += this.data[i][j] * out.gradients[i][j];
                        }
                        else if(multiplicator.cols==1){
                            this.gradients[i][j] += multiplicator.data[i][0] * out.gradients[i][j];
                            multiplicator.gradients[i][0] += this.data[i][j] * out.gradients[i][j];
                        }
                        else {
                            throw new IllegalArgumentException("Unable to broadcast. Multiplicator can be either row or column vector.");
                        }
                    }

                }

        };
        return out;
    }


        public Tensor hadamard(Tensor other){
            Tensor out=new Tensor (MathHelper.hadamard(this.data,other.data), Set.of(this, other), this.label+".*"+other.label);
            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=other.data[i][j]*out.gradients[i][j];
                        other.gradients[i][j]+=this.data[i][j]*out.gradients[i][j];
                    }
                }
            };
            return out;
        }



        public Tensor normRows(){
            Tensor out=new Tensor(this.data, Set.of(this),this.label+".Normalized()");
            double [] sums= new double[this.rows];
            double sum;
            for (int i=0;i<this.rows;i++){
                sums[i]=MathHelper.sum(this.data[i]);
            }
            for (int i=0;i<this.rows;i++){
                for(int j=0;j<this.cols;j++){
                    out.data[i][j]=this.data[i][j]/sums[i];
                }
            }

            out.localgradients=()->{
                for (int i=0;i<out.rows;i++){
                    for (int j=0;j<out.cols;j++){
                        this.gradients[i][j]+=1/sums[i]*out.gradients[i][j];
                    }
                }
            };

            return out;
        }



        public Tensor oneHot(int[] pos){
            //Tensor out=new Tensor(this.rows,this.cols,new HashSet<>(),"oneHot");
            this.data=MathHelper.oneHot(this.data,pos);
            return this;

        }

    public Tensor oneHot(double[] pos){
            int [] out=new int[pos.length];
            for (int i=0;i<pos.length;i++){
                out[i]=(int)Math.round(pos[i]);
            }
            //Tensor out=new Tensor(this.rows,this.cols,new HashSet<>(),"oneHot");
            this.data=MathHelper.oneHot(this.data,out);
            return this;

    }

        public Tensor randTensor(){
            this.data=MathHelper.randTensor(this.rows,this.cols);
            return this;
        }

        public Tensor ones(){
            this.data=MathHelper.ones(this.rows,this.cols);
            return this;
        }

        public Tensor zeros(){
            this.data=MathHelper.zeros(this.rows,this.cols);
            return this;
        }



        public Tensor eye(){
            //Tensor out=new Tensor(this.rows,this.cols);
            this.data=MathHelper.eye(this.rows);
            return this;
        }

        public void backward() {
            Set<Tensor> topo = new HashSet<>();
            Set<Tensor> visited = new HashSet<>();
            List<Tensor> topoList = new ArrayList<>(topo);

            buildTopo(this, topoList, visited);
            for (int i=0;i<this.rows;i++){
                for (int j=0;j<this.cols;j++)
                    if (i==j){
                        this.gradients[i][j]=1.0;
                    }

            }


            for (int i = topoList.size() - 1; i >= 0; i--) {
                Tensor t = topoList.get(i);
                t.localgradients.run();
            }
        }

        public void resetGradients() {
            Set<Tensor> topo = new HashSet<>();
            Set<Tensor> visited = new HashSet<>();
            List<Tensor> topoList = new ArrayList<>(topo);
            buildTopo(this, topoList, visited);
            //int numberOfThreads = Runtime.getRuntime().availableProcessors();
            //ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
            for (int i = topoList.size() - 1; i >= 0; i--) {
                Tensor t = topoList.get(i);
                t.gradients=new double[t.rows][t.cols];
            }

            //executorService.shutdown();

            /*
            try {
                executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            */

        }


        private void buildTopo(Tensor v, List<Tensor> topoList, Set<Tensor> visited){
            if (!visited.contains(v)){
                visited.add(v);
                for (Tensor child : v._prev){
                    buildTopo(child, topoList, visited);
                }
                topoList.add(v);

            }
        }

        public Tensor softMax(){
            Tensor out=new Tensor(this.rows,this.cols,Set.of(this),"Softmax("+this.label+")");
            out.data=MathHelper.softMax(this.data);

            double [][] jacobian=new double [this.cols][this.cols];
            //let's use threads top calculate jacobians faster
            //int numberOfThreads = Runtime.getRuntime().availableProcessors();
            //ExecutorService executorService = Executors.newFixedThreadPool(numberOfThreads);
            double [][] test=this.gradients;

            out.localgradients=()->{
                for (int m=0;m<this.rows;m++){
                    //calculate jacobian of derivatives for each sample in batch
                    double sum=MathHelper.sum(out.data[m]);
                    final int row=m;
                    //executorService.execute(() -> {
                                for (int i = 0; i < this.cols; i++) {
                                    /*
                                    for (int j = 0; j < this.cols; j++) {
                                        if (i == j) {
                                            jacobian[i][j] = out.data[row][i] * (1 - out.data[row][j]);
                                        } else {
                                            jacobian[i][j] = -out.data[row][i] * out.data[row][j];
                                        }

                                    }*/
                                    test[m][i]=(out.data[m][i]/sum)*out.gradients[m][i];
                                }

                                /*
                                for (int j = 0; j < out.gradients[0].length; j++) {
                                    this.gradients[row][j] = 0;
                                    for (int k = 0; k < jacobian[0].length; k++) {
                                        this.gradients[row][j] += out.gradients[row][k] * jacobian[k][j];
                                    }
                                }

                                 */


                }

            };
            return out;
        }

        public Tensor mse(Tensor Y){
            Tensor out=new Tensor(1,1,Set.of(this,Y),"MSE("+this.label+","+Y.label+")");
            out.data[0][0]=MathHelper.mse(this.data,Y.data);
            out.localgradients=()->{
                for (int i=0;i<this.rows;i++){
                    for (int j=0;j<this.cols;j++){
                        this.gradients[i][j]+=(2/(double)(rows*cols)) * (this.data[i][j]-Y.data[i][j]);
                        Y.gradients[i][j]+=(-2/(double)(rows*cols)) * (this.data[i][j]-Y.data[i][j]);
                    }
                }
            };

            return out;
        }

        public Tensor crossEntropyLoss(Tensor Y){

            Tensor out=new Tensor(1,1,Set.of(this,Y),"CrossEntropyLoss("+this.label+","+Y.label+")");
            out.data[0][0]=MathHelper.crossEntropyLoss(this.data,Y.data);

            out.localgradients=()->{
                for (int i = 0; i < this.rows; i++) {
                    for (int j = 0; j < this.cols; j++) {
                        this.gradients[i][j] += (-Y.data[i][j] / this.data[i][j])/this.rows;
                    }
                }

            };
            return out;
        }

    public Tensor categoricalEntropyLoss(Tensor Y){
        Tensor out=new Tensor(1,1,Set.of(this,Y),"CategoricalCrossEntropyLoss("+this.label+","+Y.label+")");
        double [][] P=new double[this.rows][this.cols];
        P=MathHelper.softMax(this.data);
        out.data[0][0]=MathHelper.crossEntropyLoss(P,Y.data);
        this.gradients=P;
        out.localgradients=()->{
            //this.gradients=sub(softMax(this.gradients),Y.data);
            for (int i=0;i<this.gradients.length;i++){
                for (int j=0;j<this.gradients[0].length;j++){
                    this.gradients[i][j]=(this.gradients[i][j]-Y.data[i][j])/this.gradients.length;
                }
            }
        };

        return out;
    }




    public Tensor tanh(){
        Tensor out=new Tensor(this.rows,this.cols,Set.of(this),"tanh("+this.label+")");
        for (int i=0;i<this.rows;i++){
            for (int j=0;j<this.cols;j++){
                out.data[i][j]=Math.tanh(this.data[i][j]);
            }
        }
        out.localgradients=()->{
          for(int i=0;i<this.rows;i++){
              for(int j=0;j<this.cols;j++){
                  this.gradients[i][j]+=(1-Math.pow(out.data[i][j],2))* out.gradients[i][j];
              }
          }
        };
        return out;
    }

    public Tensor mapValues(Tensor indexes){
            Tensor out=null;
            /*
            Tensor out=new Tensor (MathHelper.mapValues(indexes.data,this.data),Set.of(this,indexes), "mapValues("+this.label+")");
            out.localgradients=()->{
                for (int i = 0; i < indexes.rows; i++) {
                    for (int j = 0; j < indexes.cols; j++) {
                        int index = (int) indexes.data[i][j];
                        indexes.gradients[i][j]+= (index >= 0 && index < this.rows) ? out.gradients[i][j] : 0.0;
                        this.gradients[i][j] += (index >= 0 && index < this.rows) ? out.gradients[i][j] : 0.0;
                    }
                }

            };

             */
            return out;


    }





}
