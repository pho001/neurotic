import javax.sql.DataSource;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

public class Model {
    public enum architecture {
            MLP,
            WAVENET,
            RNN,
            LSTM
    }


    public static IModel create(int EmbeddingVectorSize, int inputSize, int outputSize, int[] hiddenLayersNeurons, int context, architecture arch, Dataset ds, NNConfiguration cfg,TokenizerFactory.Enc encoder){

        switch (arch) {
            case MLP -> {
                //this.topology=mlp(EmbeddingVectorSize,inputSize, outputSize,hiddenLayersNeurons,context,ds);
                return new Mlp(EmbeddingVectorSize,inputSize, outputSize,hiddenLayersNeurons,context,ds.getAlphabetSize(),cfg.epsilon,cfg.momentum,ds.getAlphabet(),encoder);
            }
            case WAVENET -> {
                return new Wavenet(EmbeddingVectorSize,inputSize,context,2,ds.getAlphabetSize(),hiddenLayersNeurons[0],ds.getAlphabetSize(),cfg.epsilon,cfg.momentum, ds.getAlphabet(),encoder);
            }
            case RNN -> {
                return new Rnn(EmbeddingVectorSize,inputSize, outputSize,200,hiddenLayersNeurons,context,ds.getAlphabetSize(),cfg.epsilon,cfg.momentum, ds.getAlphabet(),encoder);
            }

            case LSTM -> {
                return new LSTM(EmbeddingVectorSize,inputSize, outputSize,200,hiddenLayersNeurons,context,ds.getAlphabetSize(),cfg.epsilon,cfg.momentum, ds.getAlphabet(),encoder); //hiddenLayersNeurons equals to hidden states size in LSTM
            }
            case default -> {
                throw  new RuntimeException("No training mode selected");
            }
        }


    }

}
