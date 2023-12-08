import java.util.HashSet;

public interface NLayer {

    //Tensor[] call(Tensor in);
    Tensor[] call(Tensor[] in);



    public HashSet <Tensor> parameters();

    public void setTrainingMode(boolean setTrainingMode);

}
