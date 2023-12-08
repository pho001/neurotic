import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.DoubleStream;

public class Main {

    public static void main(String[] args) {


        //TODO: data initialization will be moved to MODEL
        FileHandler fh = new FileHandler("data/ceskaslova.txt"); // data/names.txt -


        List<String> set = fh.ReadFileLines();

        //just lookup table based words generation
        BigramModel bm = new BigramModel(set);
        for (int i = 0; i < 20; i++) {
            String sample = bm.generateStringFromDouble();
            System.out.println(sample);
        }


        Collections.shuffle(set);
        int context = 8;
        Dataset ds = new Dataset(set, context);
        NNConfiguration cfg = new NNConfiguration(); //this is dumb and has to be removed;
/*
        Model M=new Model(
                10,
                ds.getAlphabetSize(),
                ds.getAlphabetSize(),
                new int[]{300,200,300},
                context,
                Model.architecture.MLP,
                ds,
                cfg);

 */
        Model M = new Model(
                24,
                ds.getAlphabetSize(),
                ds.getAlphabetSize(),
                new int[]{300, 300},
                context,
                Model.architecture.WAVENET,
                ds,
                cfg);

        M.train(100000, 0.1, 32, ds, 100);
        M.generate(100);
        M.displayGraph(200);
        System.out.println("Model parameters:" + M.getParametersCount());
        System.out.println("Trainset Loss:" + M.splitLoss(Dataset.setType.TRAIN));
        System.out.println("Testset Loss:" + M.splitLoss(Dataset.setType.TEST));
    }











}