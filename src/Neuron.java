import java.util.*;

public class Neuron {
    private Value bias;

    private String activation;
    private List<Value> w;

    public Integer nInputs;
    public Neuron(Integer nInputs,String activation){
        this.nInputs=nInputs;
        this.activation=activation;
        w = new ArrayList<>();
        Random rd = new Random();
        bias= new Value(rd.nextDouble()*2-1, new HashSet<>(),"","bias");

        for (int i = 0; i < nInputs; i++) {
            w.add(new Value(rd.nextDouble()*2-1, new HashSet<>(),"","w"+i));
        }
    }



    public Value call(Value[] inputs){
        Value sum=bias;
        if (inputs.length != nInputs) {
            throw new IllegalArgumentException("Input size doesn't match neuron's weights");
        }

        for (int i = 0; i < inputs.length; i++) {
            sum = sum.add(w.get(i).mul(inputs[i]));
        }

        if (this.activation.equals("ReLU"))
            return sum.relu(sum);
        else
            return sum.tanh(sum);

    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(this.bias);
        return params;
    }

    public String toString() {
        if (activation.equals("ReLU")){
            return "ReLU";
        }
        else
            return  "tanh" + "Neuron(" + w.size() + ")";
    }


}
