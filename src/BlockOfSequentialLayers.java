import java.util.*;

public class BlockOfSequentialLayers {
    List<NLayer> layers=null;
    int layersCount=0;
    String label;
    Tensor [] outputTensor=null;
    Tensor.Join join=null;

    boolean setTrainingMode=true;



    public BlockOfSequentialLayers (List<NLayer> layers, String label){
        this.layersCount=layers.size();
        this.layers=layers;
        this.label=label;

    }

    public Set<BlockOfSequentialLayers> _prev=new HashSet<>();


    public Tensor [] call(Tensor input[]){
        //ugh, this is hack to concat outputs from previous blocks

        for (NLayer layer:this.layers){
            layer.setTrainingMode(this.setTrainingMode);
            input=layer.call(input);
        }
        this.outputTensor=input;
        return input;

    }

    public HashSet<Tensor> parameters () {
        HashSet <Tensor> params=new HashSet<>();
        for (NLayer layer:this.layers){
            params.addAll(layer.parameters());

        }

        return params;
    }

    public int getParamsCount(){
        HashSet <Tensor> params=this.parameters();
        int sum=0;
        for (Tensor p:params){
            sum+=p.rows*p.cols;
        }
        return sum;
    }

    public void setTrainingMode(boolean True){
        this.setTrainingMode=true;
    }

    public BlockOfSequentialLayers setParent(Set<BlockOfSequentialLayers> _prev){
        this._prev=new HashSet<>(_prev);
        return this;
    }



    public void buildTopo(BlockOfSequentialLayers lastBlock, List<BlockOfSequentialLayers> topoList, Set<BlockOfSequentialLayers> visited){
        if (!visited.contains(lastBlock)){
            visited.add(lastBlock);
            for (BlockOfSequentialLayers parent : lastBlock._prev){
                buildTopo(parent, topoList, visited);
            }
            topoList.add(lastBlock);

        }
    }

    public List<BlockOfSequentialLayers> buildTopo(){
        Set<BlockOfSequentialLayers> topo = new HashSet<>();
        Set<BlockOfSequentialLayers> visited = new HashSet<>();
        List<BlockOfSequentialLayers> topoList = new ArrayList<>(topo);
        buildTopo(this,topoList,visited);
        return topoList;
    }

    public void setConCat(Tensor.Join join){
        this.join=join;
    }


}
