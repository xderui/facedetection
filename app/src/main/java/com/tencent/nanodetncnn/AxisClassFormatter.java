package com.tencent.nanodetncnn;

import com.github.mikephil.charting.charts.BarLineChartBase;
import com.github.mikephil.charting.components.AxisBase;
import com.github.mikephil.charting.formatter.IAxisValueFormatter;

public class AxisClassFormatter implements IAxisValueFormatter {

    private final BarLineChartBase<?> chart;
    private final String[] emotions = new String[]{"neutral", "happiness","surprise", "sadness",
            "anger", "disgust", "fear"};
    public AxisClassFormatter(BarLineChartBase<?> chart) {
        this.chart=chart;
    }

    @Override
    public String getFormattedValue(float value, AxisBase axis) {
        int id=(int)value;
        return emotions[id];
    }

    @Override
    public int getDecimalDigits() {
        return 1;
    }
}
