import javax.sql.DataSource;
import java.util.*;

public class Dataset {
    private String alphabet;
    private List<String> train;
    private List<String> dev;
    private List<String> val;

    public List<String> wholeSet;
    public List<String> trainSet;
    public List<String> devSet;
    public List<String> testSet;

    private double devRatio=0.1;
    private double testRatio=0.1;
    private double trainRatio=0.8;

    public enum setType{
        TRAIN,
        DEV,
        TEST,
        ALL
    }

    public int context;


    public Dataset(List <String> lines, int context){
        int bottomIndex=0;
        int upperIndex=lines.size();
        List<String> out = new ArrayList<>();
        this.context=context;


        for (int i=bottomIndex;i<upperIndex;i++){
            out.add(lines.get(i));
        }
        this.alphabet="."+uniqueCharacters(lines);
        this.wholeSet=setNgrams(out,context);

        this.devSet=constructSet(setType.DEV);
        this.trainSet=constructSet(setType.TRAIN);
        this.testSet=constructSet(setType.TEST);;

    }

    public List<String> setNgrams(List <String> lines,int context){

        List<String> ngrams = new ArrayList<>();

        for (String line : lines) {
            String[] words = line.split(" ");
            for (String word : words) {
                //word = word.toLowerCase();
                String token="";
                for (int i = 0; i < context; i++) {
                    token += ".";
                }
                //word = word.replace(word, token + word + token);
                word = word.replace(word, token + word + ".");



                for (int i = 0; i < word.length()-context; i++) {
                    ngrams.add(word.substring(i, i + context+1));
                }
            }

        }

        return ngrams;

    }




    public List<String> getTrainingSet() {
        return this.trainSet;
    }
    public List<String> getTestSet() {
        return this.testSet;
    }
    public List<String> getDevSet() {
        return this.trainSet;
    }





    private String uniqueCharacters(List<String> words){
        Set<Character> unique = new TreeSet<>();
        for (String word:words){
            for (char c:word.toCharArray()) {
                unique.add(c);
            }
        }
        String output="";
        for (char character:unique){
            output+=character;
        }
        return output;

    }

    public String itostr (int[] vector){
        String generated="";
        for (int i=0;i<vector.length;i++){
            generated=generated + this.alphabet.charAt(vector[i]);

        }
        return generated;
    }

    public int strtoi (char c){
        int pos=-1;
        for (int i=0;i<this.alphabet.length();i++){
            if (this.alphabet.charAt(i) == c) {
                pos = i;
                break;
            }

        }
        return pos;
    }
    public int getAlphabetSize(){
        return this.alphabet.length();
    }

    public String getAlphabet(){ return this.alphabet;}

    public List<String> giveMeRandomBatch(setType which,int batchSize){
        List<String> source=null;
        switch (which) {
            case TRAIN :
                source=this.trainSet;
                break;
            case TEST  :
                source=this.testSet;
                break;
            case DEV :
                source=this.devSet;
                break;
        }

        Random rand=new Random();
        List<String> out=new ArrayList<>();

        for (int i=0;i< batchSize;i++){
            int randIndex=rand.nextInt(source.size());
            out.add(source.get(i));
        }
        return out;
    }

    public List<String> giveMeBatch(int batchSize, int startIndex, setType which){
        List<String> source=null;
        switch (which) {
            case TRAIN :
                source=this.trainSet;
                break;
            case TEST  :
                source=this.testSet;
                break;
            case DEV :
                source=this.devSet;
                break;
        }

        List<String> out=new ArrayList<>();

        for (int i=startIndex;i< startIndex+batchSize;i++){
            out.add(source.get(i));
        }
        return out;
    }

    public List<String> constructSet(setType which){
        int bottomIndex=0;
        int maxIndex=this.wholeSet.size();
        int upperIndex=0;

        switch (which) {
            case TRAIN :
                upperIndex=(int)Math.round(maxIndex*trainRatio);
            break;
            case TEST  :
                upperIndex=(int)Math.round(maxIndex*(trainRatio+testRatio));
                bottomIndex=(int)Math.round(maxIndex*(trainRatio));

            break;
            case DEV :
                bottomIndex=(int)Math.round(maxIndex*(trainRatio+testRatio));
            break;
        }

        List<String> out=new ArrayList<>();
        for (int i=bottomIndex;i<upperIndex;i++){
                out.add(this.wholeSet.get(i));
        }
        return out;
    }
    public List<String> giveMeSet(setType which){
        List<String> source=null;
        switch (which) {
            case TRAIN :
                source=this.trainSet;
                break;
            case TEST  :
                source=this.testSet;
                break;
            case DEV :
                source=this.devSet;
                break;
        }

        return source;
    }



}
