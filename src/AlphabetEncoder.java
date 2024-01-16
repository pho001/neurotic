import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class AlphabetEncoder implements  IEncoder{

    List<Character>   lookup;
    public AlphabetEncoder (List<Character> lookup){

        this.lookup=lookup;
    }


    public AlphabetEncoder(String charSet){
        this.lookup=new ArrayList<>();
        for (int i=0;i<charSet.length();i++){
            lookup.add(charSet.charAt(i));
        }
    }





    @Override
    public double[] encode(char[] input) {
        double[] output=new double [input.length];
        for (int i=0;i<input.length;i++){
            output[i]=lookup.indexOf(input[i]);
        }
        return output;
    }

    @Override
    public char[] decode(double[] input) {
        char[] output=new char[input.length];
        for (int i=0;i<input.length;i++){
            output[i]=lookup.get((int)input[i]);
        }
        return output;
    }
}
