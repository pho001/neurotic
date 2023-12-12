import java.util.HashSet;

public class EmbeddingLayer implements NLayer{
    Tensor [] out=null;
    Tensor EMB;

    int vocabSize;
    int contextLength;


    public EmbeddingLayer(int vocabSize, int embVecSize, int contextLength){
        //for (int i=0;i<contextLength;i++){
            this.EMB=new Tensor(vocabSize,embVecSize,  new HashSet<>(),"EMB").randTensor();
            this.EMB.label="EMB";
            this.vocabSize=vocabSize;


        //}
        this.contextLength=contextLength;
    }

    public Tensor [] call (Tensor [] input){
        this.out=new Tensor[input.length];

        if (contextLength!=input.length){
            throw new ArithmeticException("Embedding vector dimension doesn't match input data.");
        }
        /*
        if (contextLength%2!=0){
            throw new ArithmeticException("Context length must be even number.");
        }

         */

        for (int depth=0;depth< input.length;depth++){
            //for (int i=0;i<this.contextLength;i++) {
                Tensor X = new Tensor(input[depth].data, new HashSet<>(), "X" + depth);
            //}
                this.out[depth]=X.mul(EMB);

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
