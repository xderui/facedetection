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

#include "retinaface.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

class RetinafaceFocus: public ncnn::Layer
{
public:
    RetinafaceFocus()
    {
        one_blob_only = true;
    }

    virtual  int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
        {
            return -100;
        }

        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float *outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }
                ptr += w;
            }
        }

        return 0;
    }
};


//
//class PPlcnetFocus: public ncnn::Layer
//{
//public:
//    PPlcnetFocus()
//    {
//        one_blob_only = true;
//    }
//
//    virtual  int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
//    {
//        int w = bottom_blob.w;
//        int h = bottom_blob.h;
//        int channels = bottom_blob.c;
//
//        int outw = w / 2;
//        int outh = h / 2;
//        int outc = channels * 4;
//
//        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
//        if (top_blob.empty())
//        {
//            return -100;
//        }
//
//        for (int p = 0; p < outc; p++)
//        {
//            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
//            float *outptr = top_blob.channel(p);
//
//            for (int i = 0; i < outh; i++)
//            {
//                for (int j = 0; j < outw; j++)
//                {
//                    *outptr = *ptr;
//
//                    outptr += 1;
//                    ptr += 2;
//                }
//                ptr += w;
//            }
//        }
//
//        return 0;
//    }
//};


DEFINE_LAYER_CREATOR(RetinafaceFocus);

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


Retinaface::Retinaface() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Retinaface::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    retinaface.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    retinaface.opt = ncnn::Option();
//    pplcnet.opt = ncnn::Option();

#if NCNN_VULKAN
    retinaface.opt.use_vulkan_compute = use_gpu;
//    pplcnet.opt.use_vulkan_compute = use_gpu;
#endif
    retinaface.register_custom_layer("RetinafaceFocus", RetinafaceFocus_layer_creator);
    retinaface.opt.num_threads = ncnn::get_big_cpu_count();
    retinaface.opt.blob_allocator = &blob_pool_allocator;
    retinaface.opt.workspace_allocator = &workspace_pool_allocator;
//    pplcnet.register_custom_layer("PPlcnetFocus", PPlcnetFocus_layer_creator);
//    pplcnet.opt.num_threads = ncnn::get_big_cpu_count();
//    pplcnet.opt.blob_allocator = &blob_pool_allocator;
//    pplcnet.opt.workspace_allocator = &workspace_pool_allocator;


    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    retinaface.load_param(parampath);
    retinaface.load_model(modelpath);
    pplcnet.load_param("minixception.ncnn.param");
    pplcnet.load_model("minixception.ncnn.bin");

    target_size = _target_size;

    mean_vals[0] = 1 / 255.f;
    mean_vals[1] = 1 / 255.f;
    mean_vals[2] = 1 / 255.f;
    norm_vals[0] = 0;
    norm_vals[1] = 0;
    norm_vals[2] = 0;

    return 0;
}

int Retinaface::load(AAssetManager* mgr, const char* modeltype, int _target_size, bool use_gpu)
{
    retinaface.clear();
//    pplcnet.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    retinaface.opt = ncnn::Option();
//    pplcnet.opt = ncnn::Option();
#if NCNN_VULKAN
    retinaface.opt.use_vulkan_compute = use_gpu;
//    pplcnet.opt.use_vulkan_compute = use_gpu;
#endif
    retinaface.register_custom_layer("RetinafaceFocus", RetinafaceFocus_layer_creator);
    retinaface.opt.num_threads = ncnn::get_big_cpu_count();
    retinaface.opt.blob_allocator = &blob_pool_allocator;
    retinaface.opt.workspace_allocator = &workspace_pool_allocator;
//    pplcnet.register_custom_layer("PPlcnetFocus", PPlcnetFocus_layer_creator);
//    pplcnet.opt.num_threads = ncnn::get_big_cpu_count();
//    pplcnet.opt.blob_allocator = &blob_pool_allocator;
//    pplcnet.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];

    target_size = 640;

    sprintf(parampath, "%s.param", modeltype);
    sprintf(modelpath, "%s.bin", modeltype);

    retinaface.load_param(mgr, parampath);
    retinaface.load_model(mgr, modelpath);
    pplcnet.load_param(mgr,"minixception.ncnn.param");
    pplcnet.load_model(mgr,"minixception.ncnn.bin");

    return 0;
}





// copy from src/layer/proposal.cpp
static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat &ratios, const ncnn::Mat &scales) {
    int num_ratio = ratios.w;
    int num_scale = scales.w;

    ncnn::Mat anchors;
    anchors.create(4, num_ratio * num_scale);

    const float cx = base_size * 0.5f;
    const float cy = base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++) {
        float ar = ratios[i];

        int r_w = round(base_size / sqrt(ar));
        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++) {
            float scale = scales[j];

            float rs_w = r_w * scale;
            float rs_h = r_h * scale;

            float *anchor = anchors.row(i * num_scale + j);

            anchor[0] = cx - rs_w * 0.5f;
            anchor[1] = cy - rs_h * 0.5f;
            anchor[2] = cx + rs_w * 0.5f;
            anchor[3] = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void
generate_proposals(const ncnn::Mat &anchors, int feat_stride, const ncnn::Mat &score_blob,
                   const ncnn::Mat &bbox_blob,
                   const ncnn::Mat &landmark_blob, float prob_threshold,
                   std::vector <Object> &faceobjects) {
    int w = score_blob.w;
    int h = score_blob.h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.h;

    for (int q = 0; q < num_anchors; q++) {
        const float *anchor = anchors.row(q);

        const ncnn::Mat score = score_blob.channel(q + num_anchors);
        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

        // shifted anchor
        float anchor_y = anchor[1];

        float anchor_w = anchor[2] - anchor[0];
        float anchor_h = anchor[3] - anchor[1];

        for (int i = 0; i < h; i++) {
            float anchor_x = anchor[0];

            for (int j = 0; j < w; j++) {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold) {
                    // apply center size
                    float dx = bbox.channel(0)[index];
                    float dy = bbox.channel(1)[index];
                    float dw = bbox.channel(2)[index];
                    float dh = bbox.channel(3)[index];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0 + 1;
                    obj.rect.height = y1 - y0 + 1;

                    obj.prob = prob;

                    faceobjects.push_back(obj);
                }

                anchor_x += feat_stride;
            }

            anchor_y += feat_stride;
        }
    }
}



std::vector<int> Retinaface::detect(const cv::Mat& rgb, std::vector<Object>& faceobjects, float prob_threshold, float nms_threshold)
{

    int img_w = rgb.cols;
    int img_h = rgb.rows;

    // letterbox pad to multiple of 32
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }


    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    ncnn::Extractor ex = retinaface.create_extractor();

    ex.input("data", in);

    std::vector <Object> faceproposals;

    // stride 32
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
        ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 32;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 32.f;
        scales[1] = 16.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector <Object> faceobjects32;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob,
                           prob_threshold, faceobjects32);

        faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
    }

    // stride 16
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
        ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 16;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 8.f;
        scales[1] = 4.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector <Object> faceobjects16;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob,
                           prob_threshold, faceobjects16);

        faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
    }

    // stride 8
    {
        ncnn::Mat score_blob, bbox_blob, landmark_blob;
        ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
        ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
        ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);

        const int base_size = 16;
        const int feat_stride = 8;
        ncnn::Mat ratios(1);
        ratios[0] = 1.f;
        ncnn::Mat scales(2);
        scales[0] = 2.f;
        scales[1] = 1.f;
        ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

        std::vector <Object> faceobjects8;
        generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob,
                           prob_threshold, faceobjects8);

        faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(faceproposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(faceproposals, picked, nms_threshold);

    int face_count = picked.size();
    faceobjects.resize(face_count);


    std::vector<int>result(7,0);
    for (int i = 0; i < face_count; i++) {
        faceobjects[i] = faceproposals[picked[i]];

        // clip to image size
        float x0 = faceobjects[i].rect.x;
        float y0 = faceobjects[i].rect.y;
        float x1 = x0 + faceobjects[i].rect.width;
        float y1 = y0 + faceobjects[i].rect.height;

        x0 = std::max(std::min(x0, (float) img_w - 1), 0.f);
        y0 = std::max(std::min(y0, (float) img_h - 1), 0.f);
        x1 = std::max(std::min(x1, (float) img_w - 1), 0.f);
        y1 = std::max(std::min(y1, (float) img_h - 1), 0.f);

        // emotion classify
//        cv::Mat image_part = rgb(cv::Rect(x0,y0,x1-x0,y1-y0)); // 裁剪后的图
//        ncnn::Mat in_pack3;
//        ncnn::convert_packing(in,in_pack3,3);
//        cv::Mat img(in.h,in.w,CV_32FC1);
//        memcpy((uchar*)img.data,in_pack3.data,in.w*in.h*sizeof(float));
//        cv::Mat image_part = img(cv::Rect(x0,y0,x1-x0,y1-y0));
//        ncnn::Mat in2 = ncnn::Mat::from_pixels(image_part.data,ncnn::Mat::PIXEL_RGB2GRAY,48,48);
//        cv::Mat source2 = cv::Mat(in.h,in.w,CV_32FC3);
//        in.to_pixels(source2.data,ncnn::Mat::PIXEL_RGB);
//        cv::Mat image_part = source2(cv::Rect(x0,y0,x1-x0,y1-y0));
//        int w_ = image_part.cols;
//        int h_ = image_part.rows;
//        ncnn::Mat in2 = ncnn::Mat::from_pixels(image_part.data, ncnn::Mat::PIXEL_RGB2GRAY,48,48);
//        __android_log_print(ANDROID_LOG_INFO, "native-log", "%d %d %d",in2.c,in2.h,in2.w);
//        ex2.input("input.1",in2);
//        ncnn::Mat output;
//        ex2.extract("169",output);



//        const float *out = output.channel(0);
//        __android_log_print(ANDROID_LOG_INFO, "native-log", "%d %d %d",output.c,output.h,output.w);
//        __android_log_print(ANDROID_LOG_INFO, "native-log", "%f",out[6]);

        ncnn::Mat rgb_source = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h);


        cv::Mat imageDate(rgb_source.h, rgb_source.w, CV_8UC3);
        for(int c=0;c<3;++c) {
            for (int j = 0; j < rgb_source.h; ++j) {
                for (int k = 0; k < rgb_source.w; ++k) {
                    float t = ((float*)rgb_source.data)[k + j * rgb_source.w + c * rgb_source.h * rgb_source.w];
                    imageDate.data[(2 - c) + k * 3 + j * rgb_source.w * 3] = t;
                }
            }
        }
        cv::Rect rect(x0,y0,x1 - x0,y1-y0);
        cv::Mat crop_img = imageDate(rect);

        cv::Mat GrayImage;
        cv::cvtColor(crop_img,GrayImage,cv::COLOR_BGR2GRAY);
        cv::Mat face_img;
        cv::resize(GrayImage,face_img,cv::Size(48,48),cv::INTER_AREA);
        __android_log_print(ANDROID_LOG_INFO, "native-log", "%d",i);
        ncnn::Mat face_input = ncnn::Mat::from_pixels(face_img.data,ncnn::Mat::PIXEL_GRAY,face_img.cols,face_img.rows);
        __android_log_print(ANDROID_LOG_INFO, "native-log", "%d %d %d",face_input.c,face_input.w,face_input.h);
        const float *inp = face_input.channel(0);
        __android_log_print(ANDROID_LOG_INFO, "native-log", "%f %f %f",inp[0],inp[1],inp[2]);
        const float meanValues[1] = {0};
        const float normValues[1] = {1.0f/256.0};
//        const float meanValues[3] = {0.0f,0.0f,0.0f};
//        const float normValues[3] = {256.0f,256.0f,256.0f};
        face_input.substract_mean_normalize(meanValues, normValues);
        ncnn::Extractor ex2 = pplcnet.create_extractor();
        ex2.input("in0",face_input);
        ncnn::Mat output;
        ex2.extract(40,output);
        const float *out = output.channel(0);


        int k;
        float curmax=-1000.0;
        for(int j=0;j<7;++j){
            __android_log_print(ANDROID_LOG_INFO, "native-log", "%f ", out[j]);
            if(out[j]>curmax){
                curmax=out[j];
                k=j;
            }
        }

        faceobjects[i].rect.x = x0;
        faceobjects[i].rect.y = y0;
        faceobjects[i].rect.width = x1 - x0;
        faceobjects[i].rect.height = y1 - y0;
        faceobjects[i].label = k;

        result[k]++;

    }

    return result;

}

int Retinaface::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
//    static const char* class_names[] = {
//            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//            "hair drier", "toothbrush"
//    };
//{'surprise':0,'fear':1,'disgust':2,'happy':3,'sad':4,'angry':5,'neutral':6}
//    static const char* class_names[] = {"surprise","fear","disgust","happy","sad","angry","neutral"};
//    {0:'neutral',1:'happiness',2:'surprise',3:'sadness',4:'anger',5:'disgust',6:'fear',
//                7:'contempt',8:'unknown',9:'NF'}
    static const char* class_names[] = {"neutral", "happiness","surprise", "sadness",
                                        "anger", "disgust", "fear"};
    static const unsigned char colors[1][3] = {
            {0,0,255}
    };
//    static const unsigned char colors[19][3] = {
//            { 54,  67, 244},
//            { 99,  30, 233},
//            {176,  39, 156},
//            {183,  58, 103},
//            {181,  81,  63},
//            {243, 150,  33},
//            {244, 169,   3},
//            {212, 188,   0},
//            {136, 150,   0},
//            { 80, 175,  76},
//            { 74, 195, 139},
//            { 57, 220, 205},
//            { 59, 235, 255},
//            {  7, 193, 255},
//            {  0, 152, 255},
//            { 34,  87, 255},
//            { 72,  85, 121},
//            {158, 158, 158},
//            {139, 125,  96}
//    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 1];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb,obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    }


    return 0;
}