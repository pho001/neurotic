import java.util.ArrayList;
import java.util.List;


public class Layer {
    private List<Neuron> neurons;
    public Integer size;
    public double[] outputs;

    public Layer(Integer nInputs, Integer nSize, String activation){
        this.neurons = new ArrayList<>();
        this.size=nSize;
        for (int i=0;i<nSize;i++){
            neurons.add(new Neuron(nInputs,activation));
        }
    }

    /*
    public double[] forward(double[] inputs){

        double[] output = new double[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            output[i]=neurons.get(i).forward(inputs).data;
        }
    return output;
    }
    */


    public Value[] call(Value[] input){
        Value[] out = new Value[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            out[i] = neurons.get(i).call(input);
        }

        return out.length == 1 ? new Value[]{out[0]} : out;
    }

    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Neuron n : neurons) {
            params.addAll(n.parameters());
        }
        return params;
    }

    public String toString() {
        List<String> neuronStrings = new ArrayList<>();
        for (Neuron n : neurons) {
            neuronStrings.add(n.toString());
        }
        return "Layer of [" + String.join(", ", neuronStrings) + "]";
    }


}
