import java.util.HashSet;

public class EmbeddingLayerSqueezed implements NLayer{
    Tensor out[]=null;
    Tensor EMB;

    int contextLength;


    public EmbeddingLayerSqueezed(int vocabSize, int embVecSize, int contextLength){
        this.EMB=new Tensor(vocabSize,embVecSize,  new HashSet<>(),"EMB").randTensor();
        this.EMB.label="EMB";
        this.contextLength=contextLength;
    }
    /*
    @Override
    public Tensor [] call (Tensor input){
        this.out=new Tensor[1];
        if (contextLength!=input.rows){
            throw new ArithmeticException("Embedding vector dimension doesn't match input data");
        }
        for (int i=0;i<this.contextLength;i++){
            Tensor X=new Tensor(input.cols,EMB.rows,new HashSet<>(),"X"+i).oneHot(input.data[i]);
            if (this.out!=null){
                this.out[0]=this.out[0].join(X.dot(EMB), Tensor.Join.RIGHT);
            }
            else {
                this.out[0]=X.dot(EMB);
            }
        }

        return this.out;
    }

     */

    @Override
    public Tensor [] call (Tensor[] input){
        this.out=new Tensor[input.length];
        if (contextLength!=input[0].rows){
            throw new ArithmeticException("Embedding vector dimension doesn't match input data");
        }
        for (int depth=0;depth< input.length;depth++){
            for (int i=0;i<this.contextLength;i++){
                Tensor X=new Tensor(input[depth].cols,EMB.rows,new HashSet<>(),"X"+i).oneHot(input[depth].data[i]);
                if (this.out!=null){
                    this.out[depth]=this.out[depth].join(X.mul(EMB), Tensor.Join.RIGHT);
                }
                else {
                    this.out[depth]=X.mul(EMB);
                }
            }
        }
        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(this.EMB);
        return params;
    }

    @Override
    public void setTrainingMode(boolean setTrainingMode) {


    }
}
