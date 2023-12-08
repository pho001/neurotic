import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class Network {
    public List<Layer> layers;
    String activation;

    public Network(int nInputs, int[] nlayers, String activation){
        this.activation=activation;
        int[] allLayers=new int[nlayers.length+1];

        for (int i=0;i<allLayers.length;i++){
            if (i==0){
                allLayers[i]=nInputs;
            }
            else{
                allLayers[i]=nlayers[i-1];
            }
        }


        layers = new ArrayList<>();
        for (int i=0;i<nlayers.length;i++){
            layers.add(new Layer(allLayers[i],allLayers[i+1],activation));
        }

    }
    public Value[] call(Value[] inputs) {
        for (Layer layer : layers) {
            inputs = layer.call(inputs);
        }
        return inputs;
    }

    public Value[] call(double[] inputs) {

        Value[] fwpass=new Value[inputs.length];
        Value[] valueInputs=new Value[inputs.length];

        for (int i=0;i<inputs.length;i++){
            valueInputs[i]=new Value(inputs[i],new HashSet<>(),"","x"+i);
        }

        fwpass=this.call(valueInputs);
        return fwpass;
    }

    public Value[][] call (double[][] inputs){

        Value[][] fwpass= new Value[inputs.length][inputs[0].length];
        int i=0;
        for (double[] input:inputs){
            fwpass[i]= this.call(input);
            i++;
        }
        return fwpass;
    }


    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    public String toString() {
        List<String> layerStrings = new ArrayList<>();
        for (Layer layer : layers) {
            layerStrings.add(layer.toString());
        }
        return "MLP of [" + String.join(", ", layerStrings) + "]";
    }


}
