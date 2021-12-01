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

#ifndef HAND_H
#define HAND_H

#include <opencv2/core/core.hpp>
#include <net.h>
#include "landmark.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Point2f pts[21];
   
};

struct PalmObject
{
    float  score;
    cv::Rect rect;
    cv::Point2f  landmarks[7];
    float  rotation;

    float  hand_cx;
    float  hand_cy;
    float  hand_w;
    float  hand_h;
    cv::Point2f  hand_pos[4];

    cv::Mat trans_image;
    std::vector<cv::Point2f> skeleton;
};
struct DetectRegion
{
    float score;
    cv::Point2f topleft;
    cv::Point2f btmright;
    cv::Point2f landmarks[7];

    float  rotation;
    cv::Point2f  roi_center;
    cv::Point2f  roi_size;
    cv::Point2f  roi_coord[4];
};
struct Anchor
{
    float x_center, y_center, w, h;
};

struct AnchorsParams
{
    int input_size_width;
    int input_size_height;

    float min_scale;
    float max_scale;

    float anchor_offset_x;
    float anchor_offset_y;

    int num_layers;
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;
    std::vector<int>   strides;
    std::vector<float> aspect_ratios;

};

class Hand
{
public:
    Hand();

    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<PalmObject>& objects, float prob_threshold = 0.55f, float nms_threshold = 0.3f);

    int draw(cv::Mat& rgb, const std::vector<PalmObject>& objects);

private:

    ncnn::Net blazepalm_net;
    LandmarkDetect landmark;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    std::vector<Anchor> anchors;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // HAND_H
