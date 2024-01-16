import java.util.HashSet;

public interface IRecurrentLayer {

    Tensor[] call(Tensor[] in, int step);



    public HashSet<Tensor> parameters();

    public void setTrainingMode(boolean setTrainingMode);

}
