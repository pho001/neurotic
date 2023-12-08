import java.util.HashSet;

public class NonLinearLayer implements NLayer{

    public enum Nonlinearity{
        TANH,
    }

    Nonlinearity act;
    Tensor [] out;

    boolean setTrainingMode=true;

    public NonLinearLayer(Nonlinearity act){
        this.act=act;
        this.out=null;
    }
    /*
    @Override
    public Tensor call(Tensor in){
        switch (this.act) {
            case TANH:
                this.out=in.tanh();
                break;

        }
        return this.out;

    }

     */
    @Override
    public Tensor [] call(Tensor[] in){
        this.out=new Tensor [in.length];
        switch (this.act) {
            case TANH:
                for (int i=0;i<in.length;i++)
                {
                    this.out[i]=in[i].tanh();
                }

                break;

        }
        return this.out;

    }

    @Override
    public HashSet<Tensor> parameters (){
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }
    @Override
    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }

}
