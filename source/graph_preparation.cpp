#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<vector>
#include<dirent.h>
#include<sys/types.h>
#include<sys/stat.h>

#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#define FILTER_THRESHOLD 25

using namespace std;
using namespace cv;

void (*apply_edge_detection)(const Mat&, Mat&);
void (*apply_graph_preparation)(const Mat&, const Mat&, int);

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
    dst = Mat::zeros(src.size(), CV_8UC1);

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
// FIXME
void canny(const Mat &src, Mat &dst){
    blur(src, dst, Size(3,3));
    Canny(dst, dst, 0, 100, 3, false);
}

struct rawGraphsFiles{
    ofstream x, A, y, Y;

    void open(string root){
        string x_path = root+"/node_features.txt";
        string A_path = root+"/edges.txt";
        string y_path = root+"/graph_features.txt";
        string Y_path = root+"/label_names.txt";

        x.open(x_path.c_str(), ios::out | ios::trunc);
        A.open(A_path.c_str(), ios::out | ios::trunc);
        y.open(y_path.c_str(), ios::out | ios::trunc);
        Y.open(Y_path.c_str(), ios::out | ios::trunc);
    }

    void close(){
        x.close();
        A.close();
        y.close();
        Y.close();
    }
};

rawGraphsFiles writer;

void _4_local(const Mat& src_img, const Mat& edge_img, int label){
    // fill the matrix nxm with -1
    int n = edge_img.rows, m = edge_img.cols;
    static vector<vector<int>>nodes;

    if(nodes.size() != n || nodes[0].size() != m)
        nodes = vector<vector<int>>(n, vector<int>(m, -1));
    else for(int i=0; i<n; i++)
            for(int j=0; j<m; j++)
                nodes[i][j] = -1;

    int nodeid = 0, edge_count=0;

    for(int i=1; i<n-1; i++){
        for(int j=1; j<m-1; j++)
            if(edge_img.at<uchar>(i, j)>FILTER_THRESHOLD){
                nodes[i][j] = nodeid;
                writer.x<<i<<','<<j<<','<<to_string(src_img.at<uchar>(i, j))<<','<<to_string(xPrewitt(src_img, i, j))<<','<<to_string(yPrewitt(src_img, i, j))<<endl;
                // writer.x<<i<<','<<j<<',';
                // writer.x<<to_string(src_img.at<uchar>(i-1, j-1))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i-1, j))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i-1, j+1))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i, j-1))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i, j))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i, j+1))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i+1, j-1))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i+1, j))<<',';
                // writer.x<<to_string(src_img.at<uchar>(i+1, j+1))<<endl;
                
                if(j>0 && nodes[i][j-1] != -1){
                    writer.A<<nodes[i][j-1]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i][j-1]<<endl;
                    edge_count += 2;
                }

                if(i>0 && nodes[i-1][j] != -1){
                    writer.A<<nodes[i-1][j]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i-1][j]<<endl;
                    edge_count += 2;
                }

                nodeid += 1;
            }
    }
    writer.y<<nodeid<<','<<edge_count<<','<<label<<endl;
}

void _8_local(const Mat& src_img, const Mat& img, int label){
    // fill the matrix nxm with -1
    int n = img.rows, m = img.cols;
    static vector<vector<int>>nodes;

    if(nodes.size() != n || nodes[0].size() != m)
        nodes = vector<vector<int>>(n, vector<int>(m, -1));
    else for(int i=0; i<n; i++)
            for(int j=0; j<m; j++)
                nodes[i][j] = -1;

    int nodeid = 0, edge_count=0;

    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++)
            if(img.at<uchar>(i, j)>FILTER_THRESHOLD){
                nodes[i][j] = nodeid;

                writer.x<<i<<','<<j<<','<<to_string(src_img.at<uchar>(i, j))<<','<<to_string(xPrewitt(src_img, i, j))<<','<<to_string(yPrewitt(src_img, i, j))<<endl;

                if(j>0 && nodes[i][j-1] != -1){
                    writer.A<<nodes[i][j-1]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i][j-1]<<endl;
                    edge_count += 2;
                }

                if(i>0 && nodes[i-1][j] != -1){
                    writer.A<<nodes[i-1][j]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i-1][j]<<endl;
                    edge_count += 2;
                }

                if(i>0 && j>0 && nodes[i-1][j-1] != -1){
                    writer.A<<nodes[i-1][j-1]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i-1][j-1]<<endl;
                    edge_count += 2;
                }

                if(i>0 && j<m-1 && nodes[i-1][j+1] != -1){
                    writer.A<<nodes[i-1][j+1]<<','<<nodeid<<endl;
                    writer.A<<nodeid<<','<<nodes[i-1][j+1]<<endl;
                    edge_count += 2;
                }

                nodeid += 1;
            }
    }

    writer.y<<nodeid<<','<<edge_count<<','<<label<<endl;
}

bool hasEnding(const string &str, const string &suffix) {
    size_t n = str.length(), m = suffix.length();
    if(n<m)
        return false;
    return str.compare(n - m, m, suffix)==0;
}

void iter_dir(const string &src_dir_path, const string &dst_dir_path){
    mkdir(dst_dir_path.c_str(), ACCESSPERMS);

    struct dirent *entry;
    DIR *dir = opendir(src_dir_path.c_str());
    if (dir == NULL)
        return;

    static int labelid = 0;
    while ((entry = readdir(dir)) != NULL) {
        int type = entry->d_type;
        string name = entry->d_name;
        if(type == 4){
            // this entry represents a directory
            if(name != "." && name != ".."){
                string src_subdir_path = src_dir_path + '/' +name;
                string dst_subdir_path = dst_dir_path + '/' +name;
                writer.Y<<name<<endl;
                iter_dir(src_subdir_path, dst_subdir_path);
            }
        }
        else{
            // this entry represents a file
            if(!hasEnding(name, ".png") && !hasEnding(name, ".jpeg"))
                continue; // not a png or a jpg image
                
            string src_file_path = src_dir_path+'/'+name;
            string dst_file_path = dst_dir_path+'/'+name;
            static int file_count = 0;
            static Mat src, dst;

            src = imread(src_file_path, IMREAD_GRAYSCALE);
            apply_edge_detection(src, dst);
            apply_graph_preparation(src, dst, labelid);
            imwrite(dst_file_path, dst); // optional
            cout << "\r" <<"Processed "<< ++file_count <<" files"<<flush;
        }
    }
    labelid++;
    closedir(dir);
}

int main(int argc, const char** argv){
    const string method = argv[1];
    const string src = argv[2];

    string dst;
    int pos = src.find_last_of('/');
    if(pos<0)
        dst = argv[3]+string('/'+src+'_'+method);
    else dst = argv[3]+string('/'+src.substr(pos)+'_'+method);

    if(method == "prewitt")
        apply_edge_detection = prewitt;
    else if(method == "sobel")
        apply_edge_detection = sobel;
    else if(method == "canny")
        apply_edge_detection = canny;
    else return -1;

    apply_graph_preparation = _4_local;

    string root = argv[3]+string("/raw");
    mkdir(root.c_str(), ACCESSPERMS);

    writer.open(root);
    iter_dir(src, dst);
    writer.close();

    cout<<endl;
    return 0;
}