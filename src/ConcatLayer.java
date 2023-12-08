import java.util.HashSet;

public class ConcatLayer{
    Tensor out;

    boolean setTrainingMode=true;
    Tensor.Join join;

    public ConcatLayer(Tensor.Join join){
        this.join=join;
    }


    public Tensor call(Tensor X){

        return null;
    }
    public HashSet<Tensor> parameters (){
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }
    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }

}
