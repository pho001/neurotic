import java.util.HashSet;

public class OneDirectionalLSTMLayer implements NLayer{
    boolean useBias;
    boolean setTrainingMode=true;
    Tensor weight_forget=null;
    Tensor bias_forget=null;

    Tensor weight_input=null;
    Tensor bias_input=null;

    Tensor weight_cellState=null;
    Tensor bias_cellState=null;

    Tensor weight_output=null;
    Tensor bias_output=null;

    Tensor [] cell_state;
    Tensor [] output_gate;

    Tensor [] hidden_state;

    Tensor [] Forget;
    Tensor [] Input;
    Tensor [] Memory_state;
    int rows;
    int cols;
    int hiddenStateSize;


    public OneDirectionalLSTMLayer(boolean useBias, int hiddenStateSize){
        this.rows=rows;
        this.cols=cols;
        this.useBias=useBias;
        this.hiddenStateSize=hiddenStateSize;

    }

    @Override
    public Tensor [] call(Tensor [] X){

        if (weight_output==null){

            this.weight_input=new Tensor(X[0].cols+hiddenStateSize,hiddenStateSize, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(X[0].rows,2)));
            this.weight_input.label="Weights_input";
            this.weight_forget=new Tensor(X[0].cols+hiddenStateSize,hiddenStateSize, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(X[0].rows,2)));
            this.weight_forget.label="Weights_forget";
            this.weight_cellState=new Tensor(X[0].cols+hiddenStateSize,hiddenStateSize, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(X[0].rows,2)));
            this.weight_cellState.label="Weights_cellstate";
            this.weight_output=new Tensor(X[0].cols+hiddenStateSize,hiddenStateSize, new HashSet<>(), "Weights").randTensor().muleach(1/(Math.pow(X[0].rows,2)));
            this.weight_output.label="Weights_cellstate";
            if (this.useBias==true){
                this.bias_input= new Tensor(1, hiddenStateSize, new HashSet<>(), "Bias").zeros();
                this.bias_forget= new Tensor(1, hiddenStateSize, new HashSet<>(), "Bias").zeros();
                this.bias_cellState= new Tensor(1, hiddenStateSize, new HashSet<>(), "Bias").zeros();
                this.bias_output= new Tensor(1, hiddenStateSize, new HashSet<>(), "Bias").zeros();
            }

        }

        Forget=new Tensor[X.length];
        Input=new Tensor[X.length];
        Memory_state=new Tensor[X.length];
        cell_state=new Tensor[X.length];
        output_gate=new Tensor[X.length];
        hidden_state=new Tensor[X.length];

        for (int i=0;i<X.length;i++) {
            //hidden state forwarding
            if (i==0){
                Tensor prev=new Tensor(X[0].rows,hiddenStateSize,new HashSet<>(),"Previous hidden state").zeros();
                Tensor concat=prev.join(X[i], Tensor.Join.RIGHT);
                if (useBias) {
                    Forget[i]=concat.mul(weight_forget).addb(bias_forget).sigmoid();
                    Input[i]=concat.mul(weight_input).addb(bias_input).sigmoid();
                    Memory_state[i]=concat.mul(weight_cellState).addb(bias_cellState).tanh();
                    cell_state[i]=Input[i].hadamard(Memory_state[i]);
                    output_gate[i]=concat.mul(weight_output).addb(bias_output).sigmoid();
                }
                else{
                    Forget[i]=concat.mul(weight_forget).sigmoid();
                    Input[i]=concat.mul(weight_input).sigmoid();
                    Memory_state[i]=concat.mul(weight_cellState).tanh();
                    cell_state[i]=Input[i].hadamard(Memory_state[i]);
                    output_gate[i]=concat.mul(weight_output).sigmoid();
                }

                hidden_state[i]=output_gate[i].hadamard(cell_state[i].tanh());


            }
            else{

                Tensor concat=hidden_state[i-1].join(X[i], Tensor.Join.RIGHT);

                if (useBias){
                    Forget[i]=concat.mul(weight_forget).addb(bias_forget).sigmoid();
                    Input[i]=concat.mul(weight_input).addb(bias_input).sigmoid();
                    Memory_state[i]=concat.mul(weight_cellState).addb(bias_cellState).tanh();
                    cell_state[i]=Input[i].hadamard(Memory_state[i]);
                    output_gate[i]=concat.mul(weight_output).addb(bias_output).sigmoid();
                }
                else {
                    Forget[i]=concat.mul(weight_forget).sigmoid();
                    Input[i]=concat.mul(weight_input).sigmoid();
                    Memory_state[i]=concat.mul(weight_cellState).tanh();
                    cell_state[i]=Forget[i].hadamard(cell_state[i-1]).add(Input[i].hadamard(Memory_state[i]));
                    output_gate[i]=concat.mul(weight_output).sigmoid();
                }

            }
            hidden_state[i]=output_gate[i].hadamard(cell_state[i].tanh());

        }
        return hidden_state;
    }

    @Override
    public HashSet<Tensor> parameters() {
        HashSet <Tensor> params=new HashSet<>();
        params.add(weight_input);
        params.add(weight_forget);
        params.add(weight_cellState);
        params.add(weight_output);
        if (useBias) {
            params.add(this.bias_input);
            params.add(this.bias_forget);
            params.add(this.bias_cellState);
            params.add(this.bias_output);
        }

        return params;
    }

    @Override
    public void setTrainingMode(boolean setTrainingMode) {
        this.setTrainingMode=setTrainingMode;
    }
}
