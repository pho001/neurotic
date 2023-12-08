import cern.colt.matrix.tdouble.DoubleMatrix3D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix3D;

import java.util.HashSet;

public class BatchNormLayer implements NLayer{
    int dimension;
    double epsilon;
    double momentum;
    Tensor Running_mean=null;

    Tensor Running_var=null;

    Tensor Gamma=null;
    Tensor Beta=null;

    boolean setTrainingMode=true;

    Tensor [] out;
    public BatchNormLayer(int dimension, double epsilon, double momentum){
        this.dimension=dimension;
        this.epsilon=epsilon;
        this.momentum=momentum;
        Gamma=new Tensor(1,dimension, new HashSet<>(), "Gamma").ones();
        Beta=new Tensor(1,dimension, new HashSet<>(), "Beta").zeros();
        Running_mean=new Tensor(1,dimension,new HashSet<>(),"Running Mean");
        Running_var=new Tensor(1,dimension,new HashSet<>(),"Running Var");
    }
    /*
    @Override
    public Tensor[] call(Tensor in){

        Tensor Mean=null;
        Tensor Var=null;

        if (this.setTrainingMode){
            Mean=in.mean(Tensor.Dimension.BYCOLS);
            Var=in.variance(Mean);
            for (int i=0; i<this.dimension;i++){
                this.Running_mean.data[0][i]=this.momentum*Running_mean.data[0][i]+(1-this.momentum)*Mean.data[0][i];
                this.Running_var.data[0][i]=this.momentum*Running_var.data[0][i]+(1-this.momentum)*Var.data[0][i];
            }
        }
        else{
            Mean=this.Running_mean;
            Var=this.Running_var;
        }
        this.out[0]=in.batchNorm(this.Beta,this.Gamma,Mean.data,Var.data,epsilon);
        return this.out;
    }

     */

    public Tensor[] call(Tensor[] in){
        this.out=new Tensor[in.length];
        Tensor Mean=null;
        Tensor Var=null;
        Tensor Temp=new Tensor(in.length*in[0].rows,in[0].cols,new HashSet<>(),"Temp");
        for (int depth=0;depth<in.length;depth++){
            if (depth==0){
                Temp=in[0];
            }
            else
                Temp=Temp.join(in[depth], Tensor.Join.UNDER);
        }


        if (this.setTrainingMode){
            Mean=Temp.mean(Tensor.Dimension.BYCOLS);
            Var=Temp.variance(Mean);
            for (int i=0; i<this.dimension;i++){
                this.Running_mean.data[0][i]=this.momentum*Running_mean.data[0][i]+(1-this.momentum)*Mean.data[0][i];
                this.Running_var.data[0][i]=this.momentum*Running_var.data[0][i]+(1-this.momentum)*Var.data[0][i];
            }
        }
        else{
            Mean=this.Running_mean;
            Var=this.Running_var;
        }


        for (int depth=0;depth<in.length;depth++){
            this.out[depth]=in[depth].batchNorm(this.Beta,this.Gamma,Mean.data,Var.data,epsilon);
        }

        return this.out;
    }

    @Override
    public HashSet<Tensor> parameters (){
        HashSet <Tensor> params=new HashSet<>();
        params.add(this.Gamma);
        params.add(this.Beta);
        return params;
    }
    @Override
    public void setTrainingMode(boolean mode){
        this.setTrainingMode=mode;
    }

}
