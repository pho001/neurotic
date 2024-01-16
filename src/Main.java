import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.DoubleStream;

public class Main {

    public static void main(String[] args) {


        //TODO: data initialization will be moved to MODEL
        FileHandler fh = new FileHandler("data/names.txt");


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
        List<String> inputs=new ArrayList<>();
        List<String> labels=new ArrayList<>();
        List<String> dataset=ds.giveMeSet(Dataset.setType.TRAIN);

        for (int i=0;i<dataset.size();i++){
            String str= dataset.get(i);
            inputs.add(str.substring(0,context));
            labels.add(str.substring(context,context+1)); //last character is target
        }
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
        IModel M = Model.create(
                24,
                ds.getAlphabetSize(),
                ds.getAlphabetSize(),
                new int[]{200,200},
                context,
                Model.architecture.MLP,
                ds,
                cfg,
                TokenizerFactory.Enc.ALPHABET);


        M.train(2000, 0.0001, 64,ds, 100, OptimizerFactory.Opt.ADAM, inputs, labels);
        M.generate(100);
        M.displayGraph(200);
        System.out.println("Model parameters:" + M.getParametersCount());
        System.out.println("Trainset Loss:" + M.splitLoss(Dataset.setType.TRAIN));
        System.out.println("Testset Loss:" + M.splitLoss(Dataset.setType.TEST));
    }











}