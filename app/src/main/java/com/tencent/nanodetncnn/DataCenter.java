package com.tencent.nanodetncnn;


import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.ListView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.interfaces.datasets.IBarDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;
import com.github.mikephil.charting.utils.ColorTemplate;
import com.tencent.nanodetncnn.listviewitems.BarChartItem;
import com.tencent.nanodetncnn.listviewitems.ChartItem;
import com.tencent.nanodetncnn.listviewitems.LineChartItem;
import com.tencent.nanodetncnn.listviewitems.PieChartItem;

import java.util.ArrayList;
import java.util.List;

public class DataCenter extends Activity implements SurfaceHolder.Callback {

    ListView lv;
    Handler handler = null;
    private NcnnRetinaface ncnnretinaface = new NcnnRetinaface();
    private int current_model = 0;
    private int current_cpugpu = 0;
    private SurfaceView cameraView;
    private int facing = 1;
    public static final int REQUEST_CAMERA = 100;
    private ArrayList<ChartItem> list;
    private ChartDataAdapter cda;

    public static final int[] colors = {
            Color.rgb(192, 255, 140), Color.rgb(255, 247, 140), Color.rgb(255, 208, 140),
            Color.rgb(140, 234, 255), Color.rgb(255, 140, 157),
            Color.rgb(193, 37, 82), Color.rgb(255, 102, 0)
    };

    private final String[] emotions = new String[]{"angry", "disgust","fear", "happy",
            "sad", "surprise", "neutral"};

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_datacenter);
        lv=findViewById(R.id.listview);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview2);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);
        list = new ArrayList<>();
        cda = new ChartDataAdapter(getApplicationContext(), list);
        lv.setAdapter(cda);
        // process timely
        reload();
        handler = new Handler();
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                int[] class_num = ncnnretinaface.getResult();
                list.clear();
                //line
                ArrayList<ILineDataSet> dataSets = new ArrayList<>();
                for(int num=0;num<7;++num) {
                    ArrayList<Entry> lineentries = new ArrayList<>();
                    for (int i = 0; i < 7; ++i) {
                        lineentries.add(new BarEntry(i, class_num[num]));
                    }
                    LineDataSet lineDataSet = new LineDataSet(lineentries, emotions[num]);
                    lineDataSet.setColor(colors[num]);
                    lineDataSet.setLineWidth(2.5f);
                    lineDataSet.setCircleRadius(4.5f);
                    lineDataSet.setCircleColor(colors[num]);
                    dataSets.add(lineDataSet);
                }
                LineData lineData = new LineData(dataSets);
                list.add(new LineChartItem(lineData, getApplicationContext()));

                // bar
                ArrayList<BarEntry> barentries = new ArrayList<>();
                for(int i=0;i<7;++i){
                    barentries.add(new BarEntry(i,class_num[i]));
                }
                BarDataSet barDataSet = new BarDataSet(barentries,"test");
                barDataSet.setColors(colors);
                ArrayList<IBarDataSet> dataset = new ArrayList<>();
                dataset.add(barDataSet);
                BarData barData = new BarData(dataset);
                list.add(new BarChartItem(barData,getApplicationContext()));

                // pie
                ArrayList<PieEntry> pieEntries = new ArrayList<>();
                for(int i=0;i<7;++i){
                    pieEntries.add(new PieEntry((float)class_num[i],emotions[i]));
                }
                PieDataSet pieDataSet = new PieDataSet(pieEntries,"emotion");
                pieDataSet.setColors(colors);
                pieDataSet.setSliceSpace(2f);
                PieData pieData = new PieData(pieDataSet);

                list.add(new PieChartItem(pieData,getApplicationContext()));

                // set adapter
                cda.notifyDataSetChanged();
                handler.postDelayed(this,200);
            }
        };
        handler.postDelayed(runnable,2000);

    }


    private void reload()
    {
        boolean ret_init = ncnnretinaface.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init)
        {
            Log.e("MainActivity", "ncnnyolv5 loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        ncnnretinaface.setOutputWindow(holder.getSurface());

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED)
        {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        ncnnretinaface.openCamera(facing);
    }

    @Override
    public void onPause()
    {
        super.onPause();

        ncnnretinaface.closeCamera();
    }


    private class ChartDataAdapter extends ArrayAdapter<ChartItem> {

        ChartDataAdapter(Context context, List<ChartItem> objects) {
            super(context, 0, objects);
        }

        @NonNull
        @Override
        public View getView(int position, View convertView, @NonNull ViewGroup parent) {
            //noinspection ConstantConditions
            return getItem(position).getView(position, convertView, getContext());
        }

        @Override
        public int getItemViewType(int position) {
            // return the views type
            ChartItem ci = getItem(position);
            return ci != null ? ci.getItemType() : 0;
        }

        @Override
        public int getViewTypeCount() {
            return 3; // we have 3 different item-types
        }
    }


}
