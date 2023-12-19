import java.util.HashSet;

/*reduces last dimension of input by howMuch. Data in reduced new matrix form nlets, where n = howMuch.
In other words - data from reduced dimension are concated to previous dims.

 */
public class OneDirectionalRNNLayer implements NLayer{
    int howMuch;
    Tensor [] out= null;
    Tensor [] prev;
    boolean setTrainingMode=true;

    boolean useBias=false;
    int rows;
    int cols;

    Tensor weights_h;
    Tensor bias_h;
    public OneDirectionalRNNLayer(int hiddenSize, boolean useBias){
        this.useBias=useBias;
        this.rows=hiddenSize;
        this.cols=hiddenSize;

        this.weights_h=new Tensor(this.rows,this.cols, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(this.rows,2)));
        this.weights_h.label="Weights_h";
        if (this.useBias==true)
            this.bias_h= new Tensor(1, cols, new HashSet<>(), "Bias").zeros();

    }



    @Override
    public Tensor[] call(Tensor [] X){
        this.out=new Tensor[X.length];
        for (int i=0;i<X.length;i++) {
            //hidden state forwarding
            if (i==0){
                out[i]=X[i];
            }
            else{
                Tensor xh=out[i-1].mul(weights_h);
                if (useBias)
                    out[i]=X[i].add(xh).addb(bias_h);
                else
                    out[i]=X[i].add(xh);
            }

        }

        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(this.weights_h);
        if (useBias)
            params.add(this.bias_h);
        return params;
    }

    @Override
    public void setTrainingMode(boolean setTrainingMode) {
        this.setTrainingMode=setTrainingMode;
    }

}
