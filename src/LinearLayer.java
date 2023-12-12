import java.util.HashSet;

public class LinearLayer implements NLayer{
    boolean useBias=true;
    //Tensor [] weights=null;
    //Tensor [] bias=null;
    Tensor weights=null;
    Tensor bias=null;
    Tensor [] out=null;
    boolean setTrainingMode=true;
    int rows=0;
    int cols=0;
    int depth;



    public LinearLayer(int features_in, int features_out, boolean useBias){
        this.useBias=useBias;
        this.rows=features_in;
        this.cols=features_out;

        this.weights=new Tensor(this.rows,this.cols, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(this.rows,2)));
        this.weights.label="Weights";
        if (this.useBias==true)
            this.bias= new Tensor(1, cols, new HashSet<>(), "Bias").zeros();



    }
    /*
    @Override
    public Tensor [] call(Tensor X){
        this.out=new Tensor[1];
        if (this.useBias){
            this.out[0]=X.dot(this.weights[0]).addb(this.bias[0]);
        }
        else{
            this.out[0]=X.dot(this.weights[0]);
        }
        return this.out;
    }

     */

    @Override
    public Tensor [] call(Tensor [] X){
        /*
        if (this.weights==null)
            initWeights(X.length);
        */

        //X.length=cenotext dimension

        this.out=new Tensor[X.length];

        for (int i=0;i<X.length;i++) {
            if (this.useBias) {
                //this.out[i] = X[i].mul(this.weights[i]).addb(this.bias[i]);
                this.out[i] = X[i].mul(this.weights).addb(this.bias);
            } else {
                //this.out[i] = X[i].mul(this.weights[i]);
                this.out[i] = X[i].mul(this.weights);
            }
        }
        return this.out;
    }

    /*

    private void initWeights(int contextSize){
        this.weights=new Tensor[contextSize];
        if (this.useBias==true)
            this.bias=new Tensor[contextSize];

        for (int depth=0;depth<contextSize;depth++){
                if (this.useBias==true) {
                    this.bias[depth] = new Tensor(1, cols, new HashSet<>(), "Bias_" + depth).zeros();

                }
                //TODO: Kaiming initialization. Should be customizable.
                this.weights[depth]=new Tensor(this.rows,this.cols, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(this.rows,2)));

                this.weights[depth].label="Weights_"+depth;
        }

    }

     */


    @Override
    public HashSet <Tensor> parameters (){
       HashSet <Tensor> params=new HashSet<>();
       params.add(this.weights);
       if (useBias)
        params.add(this.bias);
        /*
       for (int i=0;i<this.weights.length;i++){
           params.add(this.weights[i]);
           if (useBias){
               params.add(this.bias[i]);
           }
       }

         */



       return params;
    }

    @Override
    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }




}
