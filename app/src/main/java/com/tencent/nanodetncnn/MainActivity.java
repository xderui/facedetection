// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.nanodetncnn;

import static java.lang.StrictMath.abs;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.Spinner;

import androidx.core.app.ActivityCompat;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.AxisBase;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.Legend.LegendForm;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.XAxis.XAxisPosition;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.components.YAxis.AxisDependency;
import com.github.mikephil.charting.components.YAxis.YAxisLabelPosition;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.formatter.IAxisValueFormatter;
import com.github.mikephil.charting.formatter.PercentFormatter;
import com.github.mikephil.charting.highlight.Highlight;
import com.github.mikephil.charting.interfaces.datasets.IBarDataSet;
import com.github.mikephil.charting.interfaces.datasets.IDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.github.mikephil.charting.listener.OnChartValueSelectedListener;
import com.github.mikephil.charting.utils.ColorTemplate;
import com.tencent.nanodetncnn.listviewitems.LineChartItem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    public static final int REQUEST_CAMERA = 100;

    private NcnnRetinaface ncnnretinaface = new NcnnRetinaface();
    private int facing = 1;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;
    String[] barstrings = new String[7];

    private SurfaceView cameraView;


    private Button timebtn;
    private Button historybtn;
    private Button transbtn;
    private RelativeLayout datatable;
    private int flag = 0;
    private BarChart barChart;
    private LineChart lineChart;
    private PieChart pieChart;
    public static final int[] colors = {
            Color.rgb(192, 255, 140), Color.rgb(255, 247, 140), Color.rgb(255, 208, 140),
            Color.rgb(140, 234, 255), Color.rgb(255, 140, 157),
            Color.rgb(193, 37, 82), Color.rgb(255, 102, 0)
    };

    private final String[] emotions = new String[]{"neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear"};

    Handler handler = null;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main3);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);


        barChart = findViewById(R.id.barChart);
        lineChart = findViewById(R.id.linechart);
        pieChart = findViewById(R.id.piechart);
        timebtn = findViewById(R.id.timedatashow);
        transbtn = findViewById(R.id.transcamera);
        historybtn = findViewById(R.id.historydatashow);

        timebtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                datatable = (RelativeLayout) findViewById(R.id.datatable);
                String text = (String) timebtn.getText();
                if(datatable.getVisibility()==View.INVISIBLE){
                    datatable.setVisibility(View.VISIBLE);
                }else{
                    datatable.setVisibility(View.INVISIBLE);
                }
            }
        });

        transbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                int new_face_id = 1 - facing;
                ncnnretinaface.closeCamera();
                ncnnretinaface.openCamera(new_face_id);
                facing = new_face_id;
            }
        });

        initBar();
//        initLine();
        initPie();
        reload();
        handler = new Handler();
        Runnable runnable = new Runnable() {
            @Override
            public void run() {

                int[] class_num = ncnnretinaface.getResult();

                //bar
                ArrayList<ClassItem> classItem = new ArrayList<>();
                for(int i=0;i<7;++i){
                    ClassItem tmp_classitem = new ClassItem();
                    tmp_classitem.id=i;
                    tmp_classitem.num=class_num[i];
                    tmp_classitem.name=emotions[i];
                    classItem.add(tmp_classitem);
                }
                Collections.sort(classItem,new ClassItem_compare());
                ArrayList<BarEntry> barentries = new ArrayList<>();
                int[] barcolors = new int[7];
                for (int i = 0; i < 7; ++i) {
                    barentries.add(new BarEntry(classItem.get(i).id, classItem.get(i).num));
                    barstrings[i]= classItem.get(i).name;
                    barcolors[i] = colors[classItem.get(i).id];
                }
                BarDataSet dataSet = new BarDataSet(barentries, "labels");
                dataSet.setColors(barcolors);
                ArrayList<IBarDataSet> datasets = new ArrayList<>();
                datasets.add(dataSet);
                BarData data = new BarData(datasets);
                barChart.setData(data);
                barChart.invalidate();

                //line
//                ArrayList<ILineDataSet> dataSets = new ArrayList<>();
//                for(int num=0;num<7;++num) {
//                    ArrayList<Entry> lineentries = new ArrayList<>();
//                    for (int i = 0; i < 7; ++i) {
//                        lineentries.add(new BarEntry(i, num));
//                    }
//                    LineDataSet lineDataSet = new LineDataSet(lineentries, emotions[num]);
//                    lineDataSet.setColor(colors[num]);
//                    lineDataSet.setLineWidth(2.5f);
//                    lineDataSet.setCircleRadius(4.5f);
//                    lineDataSet.setCircleColor(colors[num]);
//                    dataSets.add(lineDataSet);
//                }
//                LineData lineData = new LineData(dataSets);
//                lineChart.setData(lineData);
//                lineChart.invalidate();

                //pie
                ArrayList<Integer> color_id = new ArrayList<>();
                int k = 0;
                ArrayList<PieEntry> pieEntries = new ArrayList<>();
                for (int i = 0; i < 7; ++i) {
                    if (class_num[i] != 0) {
                        color_id.add(new Integer(i));
                        pieEntries.add(new PieEntry((float) class_num[i], emotions[i]));
                    }
                }
                int len=color_id.size();
                System.out.println(len);
                System.out.println("hahaha");
                int[] color = new int[len];
                for(int i=0;i<len;++i){
                    color[i]=colors[color_id.get(i)];
                    System.out.println("ok");
                    System.out.println(color[i]);
                }
                PieDataSet pieDataSet = new PieDataSet(pieEntries, "emotion");
                pieDataSet.setColors(color);
                pieDataSet.setSliceSpace(2f);
                pieDataSet.setValueTextSize(7f);
                pieDataSet.setValueTextColor(Color.BLACK);
                pieDataSet.setValueFormatter(new PercentFormatter());
                PieData pieData = new PieData(pieDataSet);
                pieChart.setData(pieData);
                pieChart.invalidate();

                handler.postDelayed(this, 1000);
            }
        };
        handler.postDelayed(runnable, 2000);
    }

    private void initBar() {
        XAxis xAxis = barChart.getXAxis();
        xAxis.setValueFormatter(new AxisClassFormatter(barChart));
        xAxis.setDrawGridLines(false);//设置x轴的表格线不显示
        xAxis.setAxisMinimum(0); //设置x轴从0开始绘画
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);
        xAxis.setGranularity(0.2f);

        YAxis leftAxis = barChart.getAxisLeft();
        leftAxis.setAxisMinimum(0); //设置y轴从0刻度开始
        leftAxis.setDrawGridLines(false); // 这里设置左侧y轴不显示表格线
        leftAxis.setAxisLineWidth(1); //设置y轴宽度
        leftAxis.setEnabled(true); //设置左侧的y轴显示
        leftAxis.setValueFormatter(new IAxisValueFormatter() {
            @Override
            public String getFormattedValue(float value, AxisBase axis) {
                String str = "";
                if (abs((int) value - value) < 1e-4) str = String.valueOf((int) value);
                return str;
            }

            @Override
            public int getDecimalDigits() {
                return 0;
            }
        });

        YAxis rightAxis = barChart.getAxisRight();
        rightAxis.setEnabled(false);

        barChart.getDescription().setEnabled(false);
        barChart.getLegend().setEnabled(false);

        ArrayList<BarEntry> entries = new ArrayList<>();
        for (int i = 0; i < 7; ++i) {
            entries.add(new BarEntry(i, 10));
        }
        BarDataSet dataSet = new BarDataSet(entries, "labels");
        dataSet.setColors(ColorTemplate.VORDIPLOM_COLORS);
        ArrayList<IBarDataSet> datasets = new ArrayList<>();
        datasets.add(dataSet);
        BarData data = new BarData(datasets);
        barChart.setData(data);

    }


    private void initPie() {
        SpannableString s = new SpannableString("情绪比例");
        s.setSpan(new RelativeSizeSpan(2.0f), 0, s.length(), 0);
        s.setSpan(new ForegroundColorSpan(ColorTemplate.getHoloBlue()), 0, s.length(), 0);

        pieChart.getDescription().setEnabled(false);
        pieChart.setCenterText(s);
        pieChart.setCenterTextSize(9f);
        pieChart.setUsePercentValues(true);

    }

    private void initLine() {
        lineChart.getDescription().setEnabled(false);
        lineChart.setDrawGridBackground(false);
        XAxis xAxis = lineChart.getXAxis();
        xAxis.setPosition(XAxisPosition.BOTTOM);
        xAxis.setDrawGridLines(false);
        xAxis.setDrawAxisLine(true);

        YAxis leftAxis = lineChart.getAxisLeft();
        leftAxis.setLabelCount(5, false);
        leftAxis.setAxisMinimum(0f); // this replaces setStartAtZero(true)

        YAxis rightAxis = lineChart.getAxisRight();
        rightAxis.setLabelCount(5, false);
        rightAxis.setDrawGridLines(false);
        rightAxis.setAxisMinimum(0f); // this replaces setStartAtZero(true)
    }

    private void reload() {
        boolean ret_init = ncnnretinaface.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init) {
            Log.e("MainActivity", "ncnnyolv5 loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        ncnnretinaface.setOutputWindow(holder.getSurface());

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        ncnnretinaface.openCamera(facing);
    }

    @Override
    public void onPause() {
        super.onPause();

        ncnnretinaface.closeCamera();
    }


    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);
        LinearLayout layout = (LinearLayout) findViewById(R.id.datashow);

        if (newConfig.orientation == this.getResources().getConfiguration().ORIENTATION_PORTRAIT) {
            layout.setRotation(-90);             // 顺时针旋转90度
        } else if (newConfig.orientation == this.getResources().getConfiguration().ORIENTATION_LANDSCAPE) {
            layout.setRotation(90);             // 顺时针旋转90度
        }
    }

}

class ClassItem{
    int id;
    int num;
    String name;
}

class ClassItem_compare implements Comparator<ClassItem>{

    @Override
    public int compare(ClassItem t1, ClassItem t2) {
        return t2.num-t1.num;
    }
}
