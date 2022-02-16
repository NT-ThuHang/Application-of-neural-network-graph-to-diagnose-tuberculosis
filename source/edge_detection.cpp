#include<iostream>
#include<string>
#include<cmath>
#include<dirent.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int xSobel(const Mat &image, int i, int j){
    return image.at<uchar>(i-1, j-1) +
                2*image.at<uchar>(i, j-1) +
                 image.at<uchar>(i+1, j-1) -
                  image.at<uchar>(i-1, j+1) -
                   2*image.at<uchar>(i, j+1) -
                    image.at<uchar>(i+1, j+1);
}
 
int ySobel(const Mat &image, int j, int i){
    return image.at<uchar>(i-1, j-1) +
                2*image.at<uchar>(i-1, j) +
                 image.at<uchar>(i-1, j+1) -
                  image.at<uchar>(i+1, j-1) -
                   2*image.at<uchar>(i+1, j) -
                    image.at<uchar>(i+1, j+1);
}

void sobel(const Mat &src, Mat &dst){
    int gx, gy, sum;
    int n = src.rows, m = src.cols;
    dst = Mat::zeros(Size(n, m), CV_8UC1);

    for(int i = 1; i < n - 1; i++){
        for(int j = 1; j < m - 1; j++){
            gx = xSobel(src, i, j);
            gy = ySobel(src, i, j);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<uchar>(i,j) = sum;
        }
    }
}

int xPrewitt(const Mat &image, int i, int j){
    return image.at<uchar>(i-1, j-1) +
                image.at<uchar>(i, j-1) +
                 image.at<uchar>(i+1, j-1) -
                  image.at<uchar>(i-1, j+1) -
                   image.at<uchar>(i, j+1) -
                    image.at<uchar>(i+1, j+1);
}
 
int yPrewitt(const Mat &image, int i, int j){
    return image.at<uchar>(i-1, j-1) +
                image.at<uchar>(i-1, j) +
                 image.at<uchar>(i-1, j+1) -
                  image.at<uchar>(i+1, j-1) -
                   image.at<uchar>(i+1, j) -
                    image.at<uchar>(i+1, j+1);
}

void prewitt(const Mat &src, Mat &dst){
    int gx, gy, sum;
    int n = src.rows, m = src.cols;
    dst = Mat::zeros(Size(n, m), CV_8UC1);

    for(int i = 1; i < n - 1; i++){
        for(int j = 1; j < m - 1; j++){
            gx = xPrewitt(src, i, j);
            gy = yPrewitt(src, i, j);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<uchar>(i,j) = sum;
        }
    }
}

void canny(const Mat &src, Mat &dst){
    blur(src, dst, Size(3,3));
    Canny(dst, dst, 0, 100, 3, false);
}

void (*apply_edge_detection)(const Mat&, Mat&);

bool hasEnding(const string &str, const string &suffix) {
    size_t n = str.length(), m = suffix.length();
    if(n<m)
        return false;
    return str.compare(n - m, m, suffix)==0;
}

void iter_dir(const string &src_dir_path,const string &dst_dir_path){
    mkdir(dst_dir_path.c_str(), ACCESSPERMS);

    struct dirent *entry;
    DIR *dir = opendir(src_dir_path.c_str());
    if (dir == NULL)
        return;

    while ((entry = readdir(dir)) != NULL) {
        int type = entry->d_type;
        string name = entry->d_name;
        if(type == 4){
            // this entry is a directory
            if(name != "." && name != ".."){
                string src_subdir_path = src_dir_path + '/' +name;
                string dst_subdir_path = dst_dir_path + '/' +name;
                iter_dir(src_subdir_path, dst_subdir_path);
            }
        }
        else{
            // this entry is a file
            if(!hasEnding(name, ".png"))
                continue; // not an png image
                
            string src_file_path = src_dir_path+'/'+name;
            string dst_file_path = dst_dir_path+'/'+name;
            static int file_count = 0;
            static Mat src, dst;

            src = imread(src_file_path, IMREAD_GRAYSCALE);
            apply_edge_detection(src, dst);
            imwrite(dst_file_path, dst);
            cout << "\r" <<"Processed "<< ++file_count <<" files"<<flush;
        }
    }
    closedir(dir);
}

int main(int argc, const char** argv){
    string method = argv[1];
    string src = argv[2];
    string dst = src+'_'+method;

    if(method == "prewitt")
        apply_edge_detection = prewitt;
    else if(method == "sobel")
        apply_edge_detection = sobel;
    else if(method == "canny")
        apply_edge_detection = canny;
    else return -1;
    
    iter_dir(src, dst);
    cout<<endl;
    return 0;
}