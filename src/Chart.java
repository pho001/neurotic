import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class Chart {

    private double[] data;
    int series=0;
    XYChart chart;
    public Chart(String title, String xAxisLabel, String yAxisLabel) {
        chart = new XYChartBuilder().width(600).height(400).title(title).xAxisTitle(xAxisLabel).yAxisTitle(yAxisLabel).build();

    }

    public void addSeries(double[] data) {

        chart.addSeries("Series "+series, range(1, data.length), data).setMarker(SeriesMarkers.NONE);
        this.series++;
    }

    public void display(){
        new SwingWrapper<>(chart).displayChart();
    }

    private double[] range(int start, int end) {
        double[] array = new double[end - start + 1];
        for (int i = 0; i < array.length; i++) {
            array[i] = start + i;
        }
        return array;
    }

}