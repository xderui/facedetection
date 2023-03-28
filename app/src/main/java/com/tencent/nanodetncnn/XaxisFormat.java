package com.tencent.nanodetncnn;

import com.github.mikephil.charting.charts.BarLineChartBase;
import com.github.mikephil.charting.components.AxisBase;
import com.github.mikephil.charting.formatter.IAxisValueFormatter;

public class XaxisFormat implements IAxisValueFormatter {
    private final String[] emotion_dict = {"angry", "disgust","fear", "happy",
            "sad", "surprise", "neutral"};
    private final BarLineChartBase<?> chart;

    public XaxisFormat(BarLineChartBase<?> chart) {
        this.chart = chart;
    }

    @Override
    public String getFormattedValue(float value, AxisBase axis) {
        System.out.println((int)value);
        return emotion_dict[(int)value];
    }

    @Override
    public int getDecimalDigits() {
        return 1;
    }
}
