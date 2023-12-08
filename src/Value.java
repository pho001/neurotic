import java.util.*;
import java.util.function.Consumer;

public class Value {
    public double data;
    public double gradient;
    private String operator;

    private Set<Value> _prev;

    private Runnable localgradients;

    private String label;

    public Value(double data, Set<Value> _children, String _operator,String label){
        this.data=data;
        this.gradient=0;
        this._prev=new HashSet<>(_children);
        this.operator=_operator;
        this.localgradients=()->{};
        this.label=label;
    }

    public Value add(Value other){
        Value out=new Value(this.data+other.data,Set.of(this, other),"+",this.label+"+"+other.label);
        out.localgradients=()->{
            this.gradient+=1*out.gradient;
            other.gradient+=1*out.gradient;
        };

        return out;
    }
    public Value sub(Value other){
        Value out=new Value(this.data-other.data,Set.of(this, other),"-",this.label+"-"+other.label);
        out.localgradients=()->{
            this.gradient+=-1*out.gradient;
            other.gradient+=-1*out.gradient;
        };

        return out;
    }

    public Value mse(Value exp){
        Value out=new Value(Math.pow(this.data-exp.data,2),Set.of(this, exp),"MSE","MSE("+this.label+","+exp.label+")");
        out.localgradients=()->{
            this.gradient+=2*(this.data-exp.data)*out.gradient;
            exp.gradient+=-2*(this.data-exp.data)*out.gradient;
        };

        return out;
    }

    public Value mul(Value other){
        Value out=new Value(this.data*other.data,Set.of(this, other),"*",this.label+"*"+other.label);
        out.localgradients=()->{
            this.gradient+=other.data*out.gradient;
            other.gradient+=this.data* out.gradient;
        };

        return out;
    }

    public Value pow(int exponent){
        Value out=new Value(Math.pow(this.data,exponent),Set.of(this),"ˆ",this.label+"ˆ"+exponent);
        out.localgradients=()->{
            this.gradient += (exponent*Math.pow(this.data,exponent-1))*out.gradient;
        };
        return out;
    }

    public Value relu(Value self){
        double data = self.data>0?self.data:0;
        Value out= new Value(data,Set.of(self),"ReLU","ReLU");
        out.label="ReLU";
        return out;
    }

    public Value tanh(Value self) {
        Value out=new Value(Math.tanh(self.data), Set.of(self), "tanh","tanh("+this.label+")");
        out.localgradients=()->{
            this.gradient+=(1-Math.pow(out.data,2))* out.gradient;
        };

        return out;
    }

    public void backward() {
        Set<Value> topo = new HashSet<>();
        Set<Value> visited = new HashSet<>();
        List<Value> topoList = new ArrayList<>(topo);

        buildTopo(this, topoList, visited);
        this.gradient=1;

        for (int i = topoList.size() - 1; i >= 0; i--) {
            Value v = topoList.get(i);
            v.localgradients.run();
        }


    }
    private void buildTopo(Value v, List<Value> topoList, Set<Value> visited){
        if (!visited.contains(v)){
            visited.add(v);
            for (Value child : v._prev){
                buildTopo(child, topoList, visited);
            }
            topoList.add(v);

        }
    }

    public static double[][] valueToDouble(Value [][] data){
        int rows=data.length;
        int cols=data[0].length;
        double[][] doubleData=new double[rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                doubleData[i][j]=data[i][j].data;
            }
        }

        return doubleData;
    }




    }