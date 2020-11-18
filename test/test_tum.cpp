#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vs_common/vs_common.h>
#include <flame/flame.h>
#include <flame/utils/image_utils.h>
#include <flame/utils/stats_tracker.h>
#include <flame/utils/load_tracker.h>

void setParamTUM(flame::Params& params);

struct CamPose {
    Eigen::Vector3d pos;
    Eigen::Vector4d quat;  // w,x,y,z
};

CamPose weightSum(double k1, const CamPose& p1, double k2, const CamPose& p2) {
    CamPose res;
    res.pos = k1 * p1.pos + k2 * p2.pos;
    res.quat = vs::weightMeanQuat(p1.quat, p2.quat, k1, k2);
    return res;
};

int main(int argc, char const *argv[])
{
    const char* data_dir = "/home/symao/data/tum_rgbd/rgbd_dataset_freiburg1_desk";
    cv::Size img_size(640,480);

    // read groundtruth poses
    vs::TimeBuffer<CamPose> pose_buffer(weightSum);
    std::ifstream fin_gt(vs::join(data_dir, "groundtruth.txt"));
    std::string line;
    while (getline(fin_gt, line)) {
        if (line.length() < 1 || line[0] == '#')
            continue;
        std::stringstream ss(line.c_str());
        double ts;
        CamPose p;
        ss >> ts >> p.pos(0) >> p.pos(1) >> p.pos(2) >> p.quat(1) >> p.quat(2) >> p.quat(3) >> p.quat(0);
        pose_buffer.add(ts, p);
        // std::cout << ts << " " << p.pos.transpose() << " " << p.quat.transpose() << std::endl;
    }

    bool resize_half = false;
    // read image and process
    cv::Mat K = (cv::Mat_<double>(3, 3) << 517.306408, 0, 318.643040, 0, 516.469215, 255.313989, 0, 0, 1);
    cv::Mat D = (cv::Mat_<double>(1, 5) << 0.262383, -0.953104, -0.005358, 0.002628, 1.163314);
    if (resize_half) {
        K(cv::Rect(0, 0, 3, 2)) *= 0.5;
        img_size /= 2;
    }

    Eigen::Matrix3f eigK;
    eigK << 517.306408, 0, 318.643040, 0, 516.469215, 255.313989, 0, 0, 1;
    Eigen::Matrix3f eigK_inv = eigK.inverse();

    flame::Params params;
    setParamTUM(params);
    std::shared_ptr<flame::Flame> sensor = std::make_shared<flame::Flame>(img_size.width, img_size.height,
                                             eigK, eigK_inv, params);;

    std::ifstream fin_rgb(vs::join(data_dir, "rgb.txt"));
    int process_idx = 0;
    while (getline(fin_rgb, line)) {
        if (line.length() < 1 || line[0] == '#')
            continue;
        printf("=====================%d=====================\n", process_idx);
        double ts;
        std::string f;
        std::stringstream ss(line.c_str());
        ss >> ts >> f;

        cv::Mat img = cv::imread(vs::join(data_dir, f), cv::IMREAD_GRAYSCALE);
        if (resize_half) {
            cv::resize(img, img, cv::Size(), 0.5, 0.5);
        }
        cv::Mat img_undistort;
        cv::undistort(img, img_undistort, K, D);

        CamPose p;
        bool ok = pose_buffer.get(ts, p);
        if (!ok) {
            printf("[ERROR]Cannot get pose at ts %f\n", ts);
            continue;
        }
        Eigen::Quaternionf quat(p.quat(0), p.quat(1), p.quat(2), p.quat(3));
        Eigen::Vector3f trans(p.pos.x(), p.pos.y(), p.pos.z());
        Eigen::Quaternionf q_flu_to_rdf(-0.5, -0.5, 0.5, -0.5);
        quat = q_flu_to_rdf * quat;
        trans = q_flu_to_rdf * trans;
        Sophus::SE3f pose(quat, trans);

        bool is_poseframe = (process_idx % 6 == 0);
        bool update_success = sensor->update(ts, process_idx, pose, img_undistort, is_poseframe);
        printf("success:%d\n", update_success);

        cv::Mat1f idepthmap;
        idepthmap = sensor->getInverseDepthMap();

        cv::Mat1f filter_idepthmap;
        sensor->getFilteredInverseDepthMap(&filter_idepthmap);

        cv::Mat depth(idepthmap.rows, idepthmap.cols, CV_32FC1);
        {
            float* ptr_depth = (float*) depth.data;
            for (int ii = 0; ii < depth.rows; ++ii) {
                for (int jj = 0; jj < depth.cols; ++jj) {
                    float idepth =  idepthmap(ii, jj);
                    *ptr_depth++ = (!std::isnan(idepth) && (idepth > 0)) ? 1.0f/ idepth : 0;
                }
            }
        }

        cv::Mat filter_depth(filter_idepthmap.rows, filter_idepthmap.cols, CV_32FC1);
        {
            float* ptr_depth = (float*) filter_depth.data;
            for (int ii = 0; ii < filter_depth.rows; ++ii) {
                for (int jj = 0; jj < filter_depth.cols; ++jj) {
                    float idepth =  filter_idepthmap(ii, jj);
                    *ptr_depth++ = (!std::isnan(idepth) && (idepth > 0)) ? 1.0f/ idepth : 0;
                }
            }
        }

        cv::Mat depth_show;
        depth.convertTo(depth_show, CV_8UC1, 50);
        cv::applyColorMap(depth_show, depth_show, cv::COLORMAP_JET);

        cv::Mat filter_depth_show;
        filter_depth.convertTo(filter_depth_show, CV_8UC1, 50);
        cv::applyColorMap(filter_depth_show, filter_depth_show, cv::COLORMAP_JET);

        cv::Mat img_debug;
        cv::cvtColor(img_undistort, img_debug, cv::COLOR_GRAY2BGR);
        cv::hconcat(img_debug, depth_show, img_debug);
        cv::hconcat(img_debug, filter_depth_show, img_debug);
        cv::imshow("depth", img_debug);
        cv::waitKey();
        // cv::Mat depth_color;
        // depth.convertTo(depth, CV_8UC1, 16);
        // applyColorMap(depth, depth_color, cv::COLORMAP_JET);
        // cv::imshow("disp", depth_color);
        // char key = cv::waitKey();
        // if (key == 27)
        //     break;
        process_idx++;
    }
}
void setParamTUM(flame::Params& params)
{
    params.omp_num_threads = 4;
    params.omp_chunk_size = 1024;

    params.do_idepth_triangle_filter = false;

    /*==================== Features Params ====================*/
    params.do_letterbox = false;
    params.min_grad_mag = 5.0;
    params.fparams.min_grad_mag = params.min_grad_mag;
    params.min_error = 100;

    params.detection_win_size = 16;
    params.zparams.win_size = 5;
    params.fparams.win_size = 5;
    params.max_dropouts = 5;
    params.zparams.epipolar_line_var = 4.0;

    /*==================== Regularizer Params ====================*/
    params.do_nltgv2 = true;
    params.adaptive_data_weights = false;
    params.rescale_data = false;
    params.init_with_prediction = true;
    params.idepth_var_max_graph = 0.01;
    params.rparams.data_factor = 0.15;
    params.rparams.step_x = 0.001;
    params.rparams.step_q = 125.0;
    params.rparams.theta = 0.25;
    params.min_height = -100000000000000.0;
    params.max_height = 100000000000000.0;
    params.check_sticky_obstacles = false;
}