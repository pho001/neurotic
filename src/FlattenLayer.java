import java.util.HashSet;

/*reduces last dimension of input by howMuch. Data in reduced new matrix form nlets, where n = howMuch.
In other words - data from reduced dimension are concated to previous dims.

 */
public class FlattenLayer implements NLayer{
    int howMuch;
    Tensor [] out= null;
    boolean setTrainingMode=true;

    public FlattenLayer(int howMuch){
        this.howMuch=howMuch;

    }
    /*
    @Override
    public Tensor[] call(Tensor input){
        //todo: implement 2d matrix reduction
        return null;
    }

     */
    @Override
    public Tensor[] call(Tensor [] input){
        if (input.length%howMuch!=0){
            if (input.length==1){
                return input;
            }
            throw new RuntimeException("Unable to reduce dimension.");
        }
        Tensor prev=null;
        this.out= new Tensor[input.length/this.howMuch];
        int depth=input.length;
        int k=0;
        for (int i=0;i<depth;i++){
            if(i%this.howMuch==0){
                for (int j=0;j<this.howMuch;j++){
                    if (j==0){
                        this.out[k]=input[j+i];
                    }
                    else {
                        //this.out[k]=this.out[k].join(input[j], Tensor.Join.RIGHT);
                        this.out[k]=this.out[k].join(input[j+i], Tensor.Join.RIGHT);

                    }
                }
                k++;
            }
        }
        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        return params;
    }

    @Override
    public void setTrainingMode(boolean setTrainingMode) {
        this.setTrainingMode=setTrainingMode;
    }


}
