import java.io.*;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;


public class FileHandler {
    private String fileName;
    public FileHandler(String fileName){
        this.fileName=fileName;
    }
    public List<String> ReadFileLines() {
        List <String> lines=new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            while ((line = br.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return lines;
    }

    public void SaveToJson(NNConfiguration cfg) {
        try {

            File file = new File(fileName);
            ObjectMapper objectMapper = new ObjectMapper();
            objectMapper.writeValue(file,cfg);

        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public NNConfiguration readFromJson(){
        ObjectMapper objectMapper = new ObjectMapper();
        NNConfiguration parameters;
        try {
            File file=new File(fileName);
            parameters = objectMapper.readValue(file, NNConfiguration.class);
            return parameters;
        } catch (FileNotFoundException e) {
            return new NNConfiguration();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

}
