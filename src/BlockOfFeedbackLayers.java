import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class BlockOfFeedbackLayers {
    List<NLayer> layers=null;
    int layersCount=0;
    String label;
    int feedbackCycles;

    Tensor [] outputTensor=null;

    boolean setTrainingMode=true;

    public Set<BlockOfFeedbackLayers> _prev=new HashSet<>();

    public BlockOfFeedbackLayers(List<NLayer> layers, String label, int cycles){
        this.layersCount=layers.size();
        this.layers=layers;
        this.label=label;
        this.feedbackCycles=cycles;
    }
    public Tensor [] call(Tensor input[]){
        //ugh, this is hack to concat outputs from previous blocks
        for (int i=0;i<feedbackCycles;i++){
            for (NLayer layer:this.layers){
                layer.setTrainingMode(this.setTrainingMode);
                input=layer.call(input);
            }
        }

        this.outputTensor=input;
        return input;

    }

    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }

    public BlockOfFeedbackLayers setParent(Set<BlockOfFeedbackLayers> _prev){
        this._prev=new HashSet<>(_prev);
        return this;
    }

}
