//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include "xblas.h"
#include "xdnn.h"
#include <sys/time.h>
#include "nms.h"
#include "yolo.h"
#include <getopt.h>
#include <dirent.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/crc.hpp>
using boost::property_tree::ptree;
using namespace cv;
using namespace std;
#define PFX "[CXDNN] "
#define EXECUTOR_MAX_BATCH_SZ 1
boost::crc_32_type crc_out;
boost::crc_32_type crc_in;

#define HUGESIZE 2048000

// Note: this is assuming V1 format (without separate width/height)

//# Pre-processing of input data for yolo
int prepareInputData(cv::Mat &in_frame,int img_h, int img_w, int img_depth, float *data_ptr, int *act_img_h, int *act_img_w)
{
	cv::Mat frame, reduced_img, dup_img;


	//# YOLO was trained with RGB, not BGR like Caffe
	cv::cvtColor(in_frame, frame, cv::COLOR_BGR2RGB);

	*act_img_h = frame.rows;
	*act_img_w = frame.cols;

	int height = frame.rows;
	int width = frame.cols;
	int channels = frame.channels();

	int newdim = (height > width) ? height : width;

	float scalew = float(width)/newdim;
	float scaleh = float(height)/newdim;

	int neww = int(img_w*scalew);
	int newh = int(img_h*scaleh);

	//std::cout << "newh, neww : " << newh << " " << neww << std::endl;

	cv::resize(frame, reduced_img, cv::Size(neww, newh));

	newdim = (newh > neww) ? newh : neww;
	int diffh = newdim - newh;
	int diffw = newdim - neww;

	float *dst1 = &data_ptr[0];
	float *dst2 = &data_ptr[img_h*img_w];
	float *dst3 = &data_ptr[(channels-1)*img_h*img_w];

	uchar *img_data = reduced_img.data;

	int idx = 0, frame_cntr = 0;

	for(int l_rows = 0; l_rows < img_h; l_rows++)
	{
		//std::cout << "row, idx, frame_cntr : " << l_rows << " " << idx << " " << frame_cntr << std::endl;
		for(int l_cols = 0; l_cols < img_w; l_cols++)
		{

			//# Fill default values
			if(
					((l_rows < (diffh/2)) || (l_rows >= (newh+diffh/2))) ||
					((l_cols < (diffw/2)) || (l_cols >= (neww+diffw/2)))

			)
			{
				dst1[idx] = 0.5;
				dst2[idx] = 0.5;
				dst3[idx] = 0.5;

				idx++;

			}
			else
			{
				dst1[idx] = (float)img_data[frame_cntr++]/255.0f;
				dst2[idx] = (float)img_data[frame_cntr++]/255.0f;
				dst3[idx] = (float)img_data[frame_cntr++]/255.0f;

				idx++;
			}
		} //l_cols
	} //l_rows

	return 0;
}
// prepareInputData

std::vector<std::string> getLabels(string fname) 
{
    ifstream f(fname.c_str());
    assert(f.good());

    std::vector<std::string> labels;
    for (string line; getline(f, line); )
        labels.push_back(line);

    return labels;
}

float sigmoid(float x)
{
	float out;
	if(x > 0)
		out = 1 / (1 + expf(-x));
	else
		out = 1 - (1 / (1 + expf(x)));

	return out;
}

void softmax(int startidx, float *inputarr, float *outputarr, int n, int stride)
{
	float sumexp = 0.0;
	float largest = std::numeric_limits<float>::min();

	for(int i = 0; i < n; i++)
		if( inputarr[startidx+i*stride] > largest)
			largest = inputarr[startidx+i*stride];

	for(int i = 0; i < n; i++)
	{
		float e = expf(inputarr[startidx+i*stride] - largest);
		sumexp += e;
		outputarr[startidx+i*stride] = e;
	}

	for(int i = 0; i < n; i++)
		outputarr[startidx+i*stride] /= sumexp;
}

//# Pre-processing of input data for yolo
int swlayers(int batch_size, float *fpgaOutput, bbox *bboxes,int in_height, int in_width,int *img_h, int *img_w, std::vector<std::string> labels,int out_w,int out_h,int out_d,cv::Mat &in_frame)
{
	static int imgcnt=0;
	int bboxplanes = 5;
	int groups = out_h*out_w;
	int coords = 4;
	int classes = 80;

	//int net_w = 224;
	//int net_h = 224;
	int net_w = in_width;
	int net_h = in_height;

	float scorethresh = 0.24;
	float iouthresh = 0.3;

	int numBoxes;

	//# Length of convolution output
	int cnt = out_d*out_h*out_w;

	int groupstride = 1;
	int batchstride = (groups)*(classes+coords+1);
	int beginoffset = (coords+1)*(out_w*out_h);
	int outsize = (out_w*out_h*(bboxplanes+classes))*bboxplanes;

	//std::cout << "outsize : " << outsize << std::endl;

	float *softmaxout;
	int nms_box_cnt;

	for(int batch_id = 0; batch_id < batch_size; batch_id++)
	{
		// int startidx = batch_id*outsize;
		int startidx = batch_id*HUGESIZE;
		softmaxout = fpgaOutput + startidx;

        	crc_out.reset(0);
	        crc_out.process_bytes(softmaxout, outsize*sizeof(float));
        	std::cout << "\n[ARK] softmaxout checksum : " << outsize << " " << crc_out.checksum() << std::endl;

		//# first activate first two channels of each bbox subgroup (n)
		for(int b = 0; b < bboxplanes; b++)
		{
			for(int r = batchstride*b; r < batchstride*b+2*groups; r++)
			//for(int r = batchstride*b; r < batchstride*b+groups; r++)
				softmaxout[r] = sigmoid(softmaxout[r]);
			for(int r = batchstride*b+groups*coords; r < batchstride*b+groups*coords+groups; r++)
				softmaxout[r] = sigmoid(softmaxout[r]);
		}


		//# Now softmax on all classification arrays in image
		for(int b = 0; b < bboxplanes; b++)
			for(int g = 0; g < groups; g++)
				softmax(beginoffset + b*batchstride+ g*groupstride, softmaxout, softmaxout, classes, groups);

		//# NMS
		int in_w = *img_w; 
		int in_h= *img_h; 
		
		nms_box_cnt = do_nms(softmaxout, cnt, in_w, in_h, net_w, net_h, out_w, out_h, bboxplanes, classes, scorethresh, iouthresh, &numBoxes, &bboxes);
				
		//# REPORT BOXES
		std :: cout << "\n\n[CXDNN] Found " <<  nms_box_cnt << " boxes" << std::endl;
		
		for(int j = 0; j < nms_box_cnt; j++)
		{
			fprintf(stdout, "\n\n[CXDNN] Obj %d: %s", j, labels[bboxes[j].classid].c_str());
			fprintf(stdout, "\n[CXDNN] Score = %f", bboxes[j].prob); 
			fprintf(stdout, "\n[CXDNN] (xlo,ylo) = (%d,%d)", bboxes[j].xlo,  bboxes[j].ylo);
			fprintf(stdout, "\n[CXDNN] (xhi,yhi) = (%d,%d)", bboxes[j].xhi,  bboxes[j].yhi);
			cv::rectangle(in_frame, cv::Point(bboxes[j].xhi,bboxes[j].yhi),cv::Point(bboxes[j].xlo,bboxes[j].ylo),Scalar(255,0,0),2,8,0);
		}
		char image_name[20];
		sprintf(image_name,"yolo_out_%d_%d.jpg",imgcnt, batch_size);
		cv::imwrite(image_name,in_frame);
		imgcnt++;
	} // batch_id

	return 0;
}

void update_app_args(std::vector<std::string> &app_args){
        app_args.push_back("--xclbin");
        app_args.push_back("--weights");
        app_args.push_back("--netcfg");
        app_args.push_back("--quantizecfg");
        app_args.push_back("--labels");
        app_args.push_back("--images");
        app_args.push_back("--in_w");
        app_args.push_back("--in_h");
        app_args.push_back("--out_w");
        app_args.push_back("--out_h");
        app_args.push_back("--out_d");
        app_args.push_back("--batch_sz");
        app_args.push_back("--image");
}

void check_arg_list(std::map<std::string ,std::string> &arg_map,std::vector<std::string> &app_args,std::vector<int> &miss_list){
    for (int i=0;i<app_args.size(); i++){
        std::map<std::string ,std::string>::iterator it=arg_map.find(app_args[i]);
        if (it == arg_map.end())
        {	
            miss_list.push_back(i);
        }else{

            std::cout <<"app_args = "<< app_args[i] << " eq value="<<it->second<<endl;
        }
    }
}

void arg_list_help(){
        cout<<"Expected arguments:"<<endl;
        cout<<" --xclbin=<MLsuite ROOT PATH>/overlaybins/1525/overlay_2.xclbin --data_dir=<MLsuite ROOT PATH>/apps/yolo/work/yolov2.caffemodel_data --cmd_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo.cmds.json --quant_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo_deploy_608_8b.json --labelfile=<MLsuite ROOT PATH>/examples/deployment_modes/coco_names.txt --in_img/--in_img_dir=<MLsuite ROOT PATH>/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=608 --in_h=608 --out_w=19 --out_h=19 --out_d=425" <<endl;
}

void ProcessArgs(int argc, char** argv,std::map<std::string ,std::string> &arg_map,std::vector<std::string> &app_args,string &xclbin,string &dataDir,string &netCfgFile,string &quantCfgFile,string &labelFile,int &height,int &width,int &out_h,int &out_w,int &out_d,int &batch_sz,string &dir_path,std::string &image_path){

        update_app_args(app_args);
        for(int i=1;i<argc;i=i+2){
                vector<string> result;
                result.push_back(argv[i]);
                //boost::split(result, argv[i], boost::is_any_of("="));
                if(result[0].compare("--help")==0){
                        arg_list_help();
                        exit(0);
                }
                result.push_back(argv[i+1]);
                std::vector<string>::iterator arg_it = std::find(app_args.begin(), app_args.end(), result[0]);
                if (arg_it != app_args.end())
                {
                        int v_indx = arg_it - app_args.begin();
                        arg_map[app_args[v_indx]]=result[1];
                }
        }
        std::vector<int>miss_list;
        check_arg_list(arg_map,app_args,miss_list);
	int opt_flag=0;
        if(miss_list.size()>0){
                for(int i=0;i<miss_list.size();i++){

			if(app_args[miss_list[i]].compare("--image")==0)
			{
				if(arg_map[app_args[5]].length()!=0){
					opt_flag=1;
					continue;
				}
			}
			else {
				if(app_args[miss_list[i]].compare("--images")==0)
        	                {
                	                if(arg_map[app_args[12]].length()!=0)
					opt_flag=1;
                        	        continue;
                        	}

			}	
                        std::cout << app_args[miss_list[i]] << " = argument not found"<<endl;
                }
                cout<<"Please check the missing arguments"<<endl;
		if(opt_flag==0){
                	arg_list_help();
	                exit(0);
		}
        }
        xclbin=arg_map[app_args[0]];
        dataDir=arg_map[app_args[1]];
        netCfgFile=arg_map[app_args[2]];
        quantCfgFile=arg_map[app_args[3]];
        labelFile=arg_map[app_args[4]];
        dir_path = arg_map[app_args[5]];
        height=stoi(arg_map[app_args[7]]);
        width=stoi(arg_map[app_args[6]]);
        out_h=stoi(arg_map[app_args[9]]);
        out_w=stoi(arg_map[app_args[8]]);
        out_d=stoi(arg_map[app_args[10]]);
        batch_sz=stoi(arg_map[app_args[11]]);
    	image_path=arg_map[app_args[12]];
}

int main(int argc, char** argv)
{
    std::map<std::string ,std::string> arg_map;
    std::vector<std::string> app_args;
    std::string img_path;
    std::string xclbin,netCfgFile,quantCfgFile,labelFile;
    int height,width,out_h,out_w,out_d,batch_sz;
    std::string dir_path_s;
    std::string dataDir; 

    // Arguments parser
    ProcessArgs(argc,argv,arg_map,app_args,xclbin,dataDir,netCfgFile,quantCfgFile,labelFile,height,width,out_h,out_w,out_d,batch_sz,dir_path_s,img_path);
    int fpgaout_actsize = 2048000*batch_sz;
    std::string dataDir_c = dataDir; 
    int fpgaOutputSize = 2048000 * batch_sz;
    ptree pt;

    std::ifstream jsonFile(netCfgFile);
    // Read compiler json and build trees structure 
    read_json(jsonFile, pt);
    std::unordered_map<std::string, std::pair<std::string, std::string> >  outputSet;
    std::unordered_map<std::string, std::string>  inputSet;
    std::set<std::string> throughputInterbufSet;
    std::vector<std::string> activeLayers;
    std::unordered_map< std::string, std::shared_ptr<XDNNCompilerOp> > compOp;

    std::string input_layer_name;
    std::string output_layer_name;

    for ( const auto & layer : pt.get_child("inputs")){

        std::string name = layer.second.get<std::string>("input_name", "");
        input_layer_name=name;
        cout<<"input name"<<name<<endl;
        for (const auto & lName : layer.second.get_child("next_layers")) {
            inputSet[name] = lName.second.get_value<std::string>();
        }
    }

    for ( const auto & layer : pt.get_child("outputs")){
        std::string name = layer.second.get<std::string>("output_name", "");
        //output_layer_name=name ;
        //cout<<"output name"<<name<<endl;
        for (const auto & tName : layer.second.get_child("previous_tensors")) {
            outputSet[name].first = tName.second.get_value<std::string>();
            output_layer_name = outputSet[name].first;
            cout<<" output_layer_name "<<output_layer_name<<endl;
        }
        for (const auto & lName : layer.second.get_child("previous_layers")) {
            outputSet[name].second = lName.second.get_value<std::string>();
        }
    } 

    XBLASHandle *handle = NULL;
    string kernelName = "kernelSxdnn_0";
    int ret = xblasCreate(handle, xclbin.c_str(), kernelName.c_str());
    assert(ret == 0);
    int depth = 3;
    int img_h, img_w;
    int batch_size = batch_sz;
    fprintf(stderr, "[CXDNN] Running image classification .........\n");
    float *input=(float*)malloc(height*width*depth*batch_size*sizeof(float));
    float *output=(float*)malloc(fpgaOutputSize*sizeof(float));
    float *inputPtr = input;
    float *outputPtr = output;
    std::unordered_map <std::string, std::vector<const float*> > input_ptrs;
    std::unordered_map <std::string, std::vector<float*> > output_ptrs;
    int layercnt = 0;

    std::vector<XBLASHandle*> handles;
    handles.push_back(handle);
    char * wgt_path=(char*)malloc(dataDir.size()+1);
    char *cnetCfgFile=(char*)malloc(netCfgFile.size() + 1); 
    char *cquantCfgFile=(char*)malloc(quantCfgFile.size() + 1); 
    strcpy(wgt_path, dataDir.c_str());
    strcpy(cnetCfgFile, netCfgFile.c_str());
    strcpy(cquantCfgFile, quantCfgFile.c_str());
    std::unordered_map<int, std::vector<std::vector<float>> >  fc_wb_map;
    // Load weights and get the executor handler for launching acceletor function
    XDNNScriptExecutor<float> *executor = (XDNNScriptExecutor<float>*)XDNNMakeScriptExecutorAndLoadWeights(&handle,handles.size(),wgt_path,cnetCfgFile,cquantCfgFile,30,0);

    // get labels
    std::vector<std::string> labels = getLabels(labelFile);
    DIR *dir_;
    const char * dir_path = dir_path_s.c_str();
    double total_latency=0.0;
    struct timeval start, end;
    if(dir_path_s.length()!=0){
        dir_ = opendir(dir_path); 
        if (dir_ == NULL) {
            std::cout<<"Failed to open the input images folder"<<std::endl;
            exit(0);
        }
        struct dirent *ent;
        int img_cnt=0;
        int status = 0;
        std::string thum="Thumbs.db";
        while ((ent = readdir(dir_)) != NULL) {
            if (ent->d_type == DT_REG) {
                std::vector<std::string> img_path_t;
                std::string image_file_path;
                if(thum.compare(ent->d_name)!=0){
                    image_file_path = dir_path_s + "/" + ent->d_name;
                }
		else{
			continue;
		}

                    std::vector<cv::Mat> in_frame;
                    cv::Mat in_frame_t;
                    {
                        float *ptr = input;
                        in_frame_t = cv::imread(image_file_path,1);
                        if(!in_frame_t.data)
                        {
                            std :: cout << "[ERROR] Image read failed - " << image_file_path << std :: endl;
                            return -1;
                        }
                        else
                        {
                            in_frame.push_back(in_frame_t);
                        }

                        status = prepareInputData(in_frame[0],height, width, depth, ptr,&img_h, &img_w);
                        if(status == 0)
                            std::cout << "[CXDNN] prepareInputData done" << std::endl;
			
                        std::vector<const float*> vec_tmp;
                        vec_tmp.push_back(ptr);

                        float *out_ptr = output;
                        std::vector<float*> vec_tmp_out;
                        vec_tmp_out.push_back(out_ptr);

                        input_ptrs[input_layer_name]=vec_tmp;
                        output_ptrs[output_layer_name]=vec_tmp_out;

                        gettimeofday(&start, 0);
                        executor->execute(input_ptrs, output_ptrs,0);
                        gettimeofday(&end, 0);
                        std::cout << "[CXDNN] output read done" << std::endl;
                    	bbox *bboxes;
			int it_one_image=1;
			float *out_ptr_per_img=out_ptr;
                    	swlayers(it_one_image,out_ptr_per_img, bboxes, height,width,&img_h, &img_w, labels,out_w,out_h,out_d,in_frame[0]);
                    }

                    total_latency = total_latency + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);	
                    img_cnt++;
            }
        }	
        std::cout << "\n\n[CXDNN] Profile total latency in usec : " << (total_latency)/img_cnt <<" avg of " <<img_cnt<<" iterations "<<std::endl;
    }else{
    		const char *image_path = img_path.c_str();
		float *ptr = input;
		int status=0;
                std::vector<cv::Mat> in_frame;
                cv::Mat in_frame_t;
                in_frame_t = cv::imread(image_path,1);
                if(!in_frame_t.data)
                {
                	std :: cout << "[ERROR] Image read failed - " << image_path << std :: endl;
                        return -1;
                }
                else
                {
                	in_frame.push_back(in_frame_t);
                }

              	status = prepareInputData(in_frame[0],height, width, depth, ptr,&img_h, &img_w);
                if(status == 0)
	                std::cout << "[CXDNN] prepareInputData done" << std::endl;
        	std::vector<const float*> vec_tmp;
                vec_tmp.push_back(ptr);
                float *out_ptr = output;
                std::vector<float*> vec_tmp_out;
                vec_tmp_out.push_back(out_ptr);
                input_ptrs[input_layer_name]=vec_tmp;
                output_ptrs[output_layer_name]=vec_tmp_out;
                gettimeofday(&start, 0);
                executor->execute(input_ptrs, output_ptrs,0);
	        gettimeofday(&end, 0);
                std::cout << "[CXDNN] output read done" << std::endl;
                bbox *bboxes;
                int it_one_image=1;
                float *out_ptr_per_img=out_ptr;
                swlayers(it_one_image,out_ptr_per_img, bboxes, height,width,&img_h, &img_w, labels,out_w,out_h,out_d,in_frame[0]);
                total_latency = total_latency + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);
        	std::cout << "\n\n[CXDNN] Profile total latency in usec : " << (total_latency)<<std::endl;	
    }

    return 0;
}
