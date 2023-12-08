import javax.sql.DataSource;
import java.util.*;

public class Dataset {
    private String alphabet;
    private List<String> train;
    private List<String> dev;
    private List<String> val;

    public int[][] wholeSet;
    public int[][] trainSet;
    public int[][] devSet;
    public int[][] testSet;

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

    public int[][] setNgrams(List <String> lines,int context){

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
        int [][] out = new int[context+1][ngrams.size()];



        for(int i=0;i<ngrams.size();i++){
            for (int j = 0; j < ngrams.get(i).length() - 1; j++) {

                out[j][i]=strtoi(ngrams.get(i).charAt(j));


            }
            out[context][i]=strtoi(ngrams.get(i).charAt(context));

        }
        return out;

    }




    public int[][] getTrainingSet() {
        return this.trainSet;
    }
    public int[][] getTestSet() {
        return this.testSet;
    }
    public int[][] getDevSet() {
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

    public double[][] giveMeRandomBatch(setType which,int batchSize){
        int [][] source=null;
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

        double [][] out=new double [source.length][batchSize];
        Random rand=new Random();
        for (int i=0;i<batchSize;i++){
            int randIndex=rand.nextInt(source[0].length);
            for (int j=0;j<source.length;j++){
                out[j][i]=source[j][randIndex];
                //out[i][j]=source[randIndex][j];
            }
        }

        return out;
    }

    public double[][] giveMeBatch(int batchSize, int startIndex, setType which){
        int [][] source=null;
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

        double [][] out=new double [source.length][batchSize];

        for (int i=0;i< source.length;i++){
            int col=0;
            for (int j=startIndex;j<startIndex+batchSize;j++){
                out[i][col]=source[i][j];
                col++;
            }
        }
        return out;
    }

    public int[][] constructSet(setType which){
        int bottomIndex=0;
        int upperIndex=this.wholeSet[0].length;
        switch (which) {
            case TRAIN :
                upperIndex=(int)Math.round(this.wholeSet[0].length*trainRatio);
            break;
            case TEST  :
                upperIndex=(int)Math.round(this.wholeSet[0].length*(trainRatio+testRatio));
                bottomIndex=(int)Math.round(this.wholeSet[0].length*(trainRatio));

            break;
            case DEV :
                bottomIndex=(int)Math.round(this.wholeSet[0].length*(trainRatio+testRatio));
            break;
        }

        int [][] out=new int [this.wholeSet.length][upperIndex-bottomIndex];
        for (int i=bottomIndex;i<upperIndex;i++){
            for (int j=0;j<this.wholeSet.length;j++){
                out[j][i-bottomIndex]=this.wholeSet[j][i];
            }
        }
        return out;
    }
    public int[][] giveMeSet(setType which){
        int [][] source=null;
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
