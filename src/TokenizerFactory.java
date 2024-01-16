import java.util.List;

public class TokenizerFactory
{
    enum Enc{
        ALPHABET

    }

    public static IEncoder create(Enc optimizer, List<Character> lookup){
        switch (optimizer){

            case ALPHABET -> {
                return new AlphabetEncoder(lookup);
            }
            default -> throw new IllegalStateException("Unexpected value: " + optimizer);
        }
    }

    public static IEncoder create(Enc optimizer, String lookup){
        switch (optimizer){

            case ALPHABET -> {
                return new AlphabetEncoder(lookup);
            }
            default -> throw new IllegalStateException("Unexpected value: " + optimizer);
        }
    }
}
