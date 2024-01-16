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

    int step;

    Tensor weights_h;
    Tensor bias_h;
    Tensor bias_i;

    Tensor [] hidden;

    Tensor weights_i;
    Tensor [] hidden_current;
    Tensor hidden_prev;
    int contextLength;

    int hiddenSize;
    public OneDirectionalRNNLayer(int hiddenSize,int contextLength, boolean useBias){
        this.useBias=useBias;
        this.step=0;
        this.weights_h=new Tensor(hiddenSize,hiddenSize, new HashSet<>(), "Weights_h").randTensor().muleach(1/(Math.pow(hiddenSize,2)));
        this.weights_h.label="Weights_h";
        this.hiddenSize=hiddenSize;
        this.hidden=new Tensor [contextLength];
        this.contextLength=contextLength;
        if (this.useBias==true)
            this.bias_h= new Tensor(1, hiddenSize, new HashSet<>(), "Bias").zeros();

    }



    @Override
    public Tensor[] call(Tensor [] X){
        if (step>contextLength-1)
            step=0;

        if (step==0){

                if (useBias){
                    this.hidden[step]=X[0].addb(bias_h).tanh();
                }
                else {
                    this.hidden[step]=X[0].tanh();
                }

            }
        else{
                if (useBias)
                    this.hidden[step]=hidden[step-1].mul(weights_h).add(X[0]).addb(bias_h).tanh();
                else
                    this.hidden[step]=hidden[step-1].mul(weights_h).add(X[0]).tanh();
        }

        Tensor [] out= new Tensor[1];
        out[0]=this.hidden[step];
        step++;
        return out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(this.weights_h);
        if (useBias) {
            params.add(this.bias_h);
        }

        return params;
    }

    @Override
    public void setTrainingMode(boolean setTrainingMode) {
        this.setTrainingMode=setTrainingMode;
    }

}
