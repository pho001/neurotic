import java.util.*;
import java.util.stream.Collectors;

public class BigramModel {
    private Map<Character, Map<Character, Integer>> bigramTable;
    private Map<Character, Map<Character, Integer>> bigramTableFrequencies;
    private Map<Character, Map<Character, Double>> bigramTableNormalized;
    private String alphabet = ".abcdefghijklmnopqrstuvwxyz.";
    public double [][] trainingTable;

    public double[][] data;
    List<String> lines;

    public BigramModel(List <String> words){
        this.alphabet="."+uniqueCharacters(words);

        this.bigramTable = new HashMap<>();
        this.data=doubleBigramTableFrequencies(words);

    }


    public double[] getRowFor(char letter, Map<Character, Map<Character, Integer>> btf){
        int n=btf.get(letter).size();
        double[] prob_vector=new double[n];
        int i=0;
        for (Map.Entry <Character,Integer> item:btf.get(letter).entrySet()){
            prob_vector[i]=item.getValue();
            i++;
        }

        return prob_vector;
    }


    public String itostr (int[] vector){
        String generated="";
        for (int i=0;i<vector.length;i++){
            generated=generated + alphabet.charAt(vector[i]);

        }
        return generated;
    }

    public int strtoi (char c){
        int pos=-1;
        for (int i=0;i<alphabet.length();i++){
            if (alphabet.charAt(i) == c) {
                pos = i;
                break;
            }

        }
        return pos;
    }


    public String generateStringFromDouble(){
        String output="";
        Random random=new Random();
        double cumul=.0;
        List<Map.Entry<String, Double>> bigrams = new ArrayList<>();
        int[] vector=MathHelper.generateMultinomialVector(1, this.data[0], random);
        String nextChar="";
        for (int i=0;i<vector.length;i++) {
            nextChar=itostr(vector);
        }
        cumul=this.data[0][vector[0]];
        bigrams.add(new AbstractMap.SimpleEntry<>("."+nextChar, this.data[0][vector[0]]));
        System.out.println("."+nextChar+" ; "+this.data[0][vector[0]]+ " ; "+cumul);


        output=nextChar;
        while (!nextChar.equals(".")){
            String prevChar=nextChar;
            vector=MathHelper.generateMultinomialVector(1, this.data[vector[0]],random);
            for (int i=0;i<vector.length;i++) {
                nextChar=itostr(vector);
            }
            double prob=this.data[strtoi(prevChar.charAt(0))][strtoi(nextChar.charAt(0))];
            cumul=cumul*this.data[strtoi(prevChar.charAt(0))][strtoi(nextChar.charAt(0))];
            output=output+nextChar;
            System.out.println(prevChar+nextChar+" ; "+prob+ " ; "+Math.log(prob)+ " ; "+Math.log(cumul));
        }
        return output;
    }

    private String uniqueCharacters(List <String> words){
        Set<Character> unique = new TreeSet<>();
        for (String word:words){
            for (char c:word.toCharArray()) {
                unique.add(Character.toLowerCase(c));
            }
        }
        String output="";
        for (char pismeno:unique){
            output+=pismeno;
        }
        return output;

    }

    public double[][] doubleBigramTableFrequencies (List<String> lines){
        double[][] out=new double[this.alphabet.length()][this.alphabet.length()];
        for (String line : lines) {
            String[] words = line.split(" ");
            for (String word : words) {
                word = word.toLowerCase();
                word= word.replace(word,"."+word+".");
                for (int i = 0; i < word.length() - 1; i++) {
                    char firstChar = word.charAt(i);
                    char secondChar = word.charAt(i + 1);
                    out[strtoi(firstChar)][strtoi(secondChar)]+=1;
                }
            }

        }
        return out;
    }

    public int[][] getTrainingSet(List<String> Lines){
        List<String> bigrams = new ArrayList<>();
        for (String line : Lines) {
            String[] words = line.split(" ");
            for (String word : words) {
                word = word.toLowerCase();
                word= word.replace(word,"."+word+".");
                for (int i = 0; i < word.length() - 1; i++) {
                    /*
                    char firstChar = word.charAt(i);
                    char secondChar = word.charAt(i + 1);
                    */
                    bigrams.add(word.substring(i,i+2));
                }
            }

        }
        int [][] trainingSet=new int[2][bigrams.size()];
        for(int i=0;i<bigrams.size();i++){
            trainingSet[0][i]=strtoi(bigrams.get(i).charAt(0));
            trainingSet[1][i]=strtoi(bigrams.get(i).charAt(1));
        }

        return trainingSet;


    }
    public int getAlphabetSize(){
        return this.alphabet.length();
    }




}
