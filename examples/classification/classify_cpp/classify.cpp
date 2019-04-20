#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>
#include "xblas.h"
#include "xdnn.h"
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <getopt.h>
#include <dirent.h>

#define PFX "[CXDNN] "
#define EXECUTOR_MAX_BATCH_SZ 2

using namespace std;

// Note: this is assuming V1 format (without separate width/height)
int readFromFile(const string &fname, string &layerName, 
		int &kernWidth, int &kernHeight, int &inc, int &outc,
		std::vector<float> &vals, bool hasHeader=true)
{
	ifstream f(fname.c_str());
	if (!f.good())
		return -1;

	if (hasHeader)
	{
		f >> layerName;
		f >> kernWidth;
		f >> inc;
		f >> outc;
	}
	vals.clear();
	while(!f.eof()){
                float val;
                f>> val;
                vals.push_back(val);
        }
	cout << PFX << "reading... " << layerName << " " << vals.size() << endl;

	return 0;
}

XDNNScriptExecutor<float>* loadExecutor(XBLASHandle *handle, const string &dataDir, const string &netCfgFile, const string &quantCfgFile, int &layercnt)
				{
	std::map<std::string, std::vector<float>*> weights;
	std::map<std::string, std::vector<float>*> bias;
	std::map<std::string, XDNNWeightParam<float> > m;
	std::vector<XBLASHandle*> handles;
	handles.push_back(handle);

	layercnt = 0;
	for (int fi=0; ; fi++)
	{
		stringstream ss;
		ss << dataDir << "/fwbqb_" << fi;
		string fname = ss.str();

		string layerName;
		int kern = -1;
		int inc = -1;
		int outc = -1;
		std::vector<float> *w = new vector<float>();

		// read weights
		int ret = readFromFile(fname, layerName, kern, kern, inc, outc, *w);
		if (ret < 0)
		{
			delete w;
			break;
		}
		weights[layerName] = w;

		// read bias
		std::vector<float> *b = new vector<float>();
		ss.str("");
		ss << dataDir << "/fwbqb_bias_" << fi;
		fname = ss.str();
		readFromFile(fname, layerName, kern, kern, inc, outc, *b);
		bias[layerName] = b;

		float *weightPtr = &((*w)[0]);
		float *biasPtr = &((*b)[0]);
		m[layerName] = { weightPtr, biasPtr, w->size(), b->size() };
		layercnt++;
	}

	//XDNNScriptExecutor<float> *executor
	//= new XDNNScriptExecutor<float>(handles, m, netCfgFile, quantCfgFile,
	//		EXECUTOR_MAX_BATCH_SZ, /*scaleB*/30, /*numImgPerBatch*/1); // lvraju
	XDNNScriptExecutor<float> *executor
	= new XDNNScriptExecutor<float>(handles, m, netCfgFile, quantCfgFile, /*scaleB*/30, /*numImgPerBatch*/0);
	return executor;
				}

/*
std::vector<float> loadInput(const string &dataDir)
{
  stringstream ss;
  ss << dataDir << "/input";
  string fname = ss.str();

  ifstream f(fname.c_str());
  assert(f.good());

  vector<float> vals;
  for (string line; getline(f, line); )
  {
    istringstream iss(line); 
    float val;
    while (iss >> val)
      vals.push_back(val);
  }

  cout << PFX << "input size " << vals.size() << endl;
  return vals;
}*/

void loadInput(const string &dataDir, float *vals)
{
	stringstream ss;
	ss << dataDir << "/input";
	string fname = ss.str();

	ifstream f(fname.c_str());
	assert(f.good());

	int idx = 0;

	for (string line; getline(f, line); )
	{
		istringstream iss(line);
		float val;
		while (iss >> val)
			vals[idx++] = (val);
	}

	// cout << PFX << "input size " << vals.size() << endl;
	//return vals;
}

//# Pre-processing of input data
int prepareInputData(cv::Mat &in_frame,int img_h, int img_w, int img_depth, float *data_ptr,int *act_img_h, int *act_img_w)
{
	cv::Mat resize_frame;

	int height = in_frame.rows;
	int width = in_frame.cols;
	int channels = in_frame.channels();

	cv::resize(in_frame, resize_frame, cv::Size(img_h, img_w));

	float *dst1 = &data_ptr[0];
	float *dst2 = &data_ptr[img_h*img_w];
	float *dst3 = &data_ptr[(channels-1)*img_h*img_w];

	uchar *img_data = resize_frame.data;

	int idx = 0, frame_cntr = 0;

	float *mean = (float*)malloc(3*sizeof(float));

	mean[0] = 104.007;
	mean[1] = 116.669;
	mean[2] = 122.679;

	for(int l_rows = 0; l_rows < img_h; l_rows++)
	{
		//std::cout << "row, idx, frame_cntr : " << l_rows << " " << idx << " " << frame_cntr << std::endl;
		for(int l_cols = 0; l_cols < img_w; l_cols++)
		{
			dst1[idx] = (float)img_data[frame_cntr++] - mean[0];
			dst2[idx] = (float)img_data[frame_cntr++] - mean[1];
			dst3[idx] = (float)img_data[frame_cntr++] - mean[2];

			idx++;
		} //l_cols
	} //l_rows

#if FILE_WRITE
	FILE *fp = fopen("input.txt", "w");
	if(fp == NULL)
	{
		fprintf(stderr, "failed to create file\n");
		return -1;
	}

	/* FILE *fp_ref = fopen("/proj/sdxapps/users/anup/xDNN_Files/input_preprocessed.txt", "r");
	if(fp_ref == NULL)
	{
		fprintf(stderr, "failed to open file\n");
		return -1;
	} */

	float float_val = 0;

	int cnt = 0;

	for(int k = 0; k < channels; k++)
		for(int i = 0; i < img_h; i++)
			for(int j = 0; j < img_w; j++)
			{
				fprintf(fp, "%f\n", data_ptr[cnt]);	
				/* fscanf(fp_ref, "%f ", &float_val);
				float diff = (float_val - data_ptr[cnt]);
				if(diff > 0.00001f)
				{
					fprintf(stderr, "mismatch at pos (h, w, depth, idx) : (%d, %d, %d, %d) ref  : %f out : %f error : %f\n", i+1, j+1, k+1, cnt, float_val, data_ptr[cnt], diff);
				} */
				cnt++;
			}
	fclose(fp);
	//fclose(fp_ref);
#endif

	return 0;
}
// prepareInputData


void softmax(std::vector<float> &input) 
{
	//std::cout << "softmax input size : " << input.size() << std::endl;
	float m = std::numeric_limits<float>::min();
	for (size_t i = 0; i < input.size(); i++)
		if (input[i] > m)
			m = input[i];

	float sum = 0.0;
	for (size_t i = 0; i < input.size(); i++)
		sum += expf(input[i] - m);

	for (size_t i = 0; i < input.size(); i++)
		input[i] = expf(input[i] - m) / sum;
}

std::vector<std::string> getLabels(string fname) 
		{
	ifstream f(fname.c_str());
	assert(f.good());

	std::vector<std::string> labels;
	for (string line; getline(f, line); )
		labels.push_back(line);

	return labels;
		}
void update_app_args(std::vector<std::string> &app_args){
        app_args.push_back("--xclbin");
        app_args.push_back("--datadir");
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
				if(app_args[i].compare("--image")==0){
					it=arg_map.find("--images");
					if(it == arg_map.end())
                                		miss_list.push_back(i);
				}else if(app_args[i].compare("--images")==0){
					it=arg_map.find("--image");
					if(it == arg_map.end())
                                		miss_list.push_back(i);
				}else{
					
                                		miss_list.push_back(i);
				}
				
                }else{

                        std::cout <<"app_args="<< app_args[i] << " eq value="<<it->second<<endl;
                }
        }
}
void arg_list_help(){
        cout<<"Expected arguments:"<<endl;
        cout<<" --xclbin=<MLsuite ROOT PATH>/overlaybins/1525/overlay_2.xclbin --data_dir=<MLsuite ROOT PATH>/apps/yolo/work/yolov2.caffemodel_data --cmd_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo.cmds.json --quant_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo_deploy_608_8b.json --labelfile=<MLsuite ROOT PATH>/examples/classification/coco_names.txt --in_img/--in_img_dir=<MLsuite ROOT PATH>/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=608 --in_h=608 --out_w=19 --out_h=19 --out_d=425" <<endl;
}

void ProcessArgs(int argc, char** argv,std::map<std::string ,std::string> &arg_map,std::vector<std::string> &app_args,string &xclbin,string &dataDir,string &netCfgFile,string &quantCfgFile,string &labelFile,int &height,int &width,int &out_h,int &out_w,int &out_d,int &batch_sz,string &dir_path,std::vector<std::string> &image_path){

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

        if(miss_list.size()>0){
                std::cout << "List of arguments are missing from command line\n" ;
                for(int i=0;i<miss_list.size();i++){
                        std::cout << app_args[miss_list[i]] << " = argument not found"<<endl;
                }
                cout<<"Please check the missing arguments"<<endl;
                arg_list_help();
                exit(0);
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
	std::map<std::string ,std::string>::iterator it=arg_map.find(app_args[12]);
	if (it!= arg_map.end())
        {
	       image_path.push_back(arg_map[app_args[12]]);
        }
	for(int i=1;i<batch_sz;i++){
		image_path.push_back(arg_map[app_args[12]]);
	}	

}

//# ResNet-50 infernce on xfdnn-zynq								
int main(int argc, char** argv)
{



	std::map<std::string ,std::string> arg_map;
        std::vector<std::string> app_args;
        std::vector<std::string>img_path;
        string xclbin,dataDir,netCfgFile,quantCfgFile,labelFile;
        int height,width,out_h,out_w,out_d,batch_sz;
        std::string dir_path_s;
        ProcessArgs(argc,argv,arg_map,app_args,xclbin,dataDir,netCfgFile,quantCfgFile,labelFile,height,width,out_h,out_w,out_d,batch_sz,dir_path_s,img_path);
	int fcout_actsize = 1000;
	int fpgaout_actsize = out_d;
	const int fpgaOutputSize = out_d * EXECUTOR_MAX_BATCH_SZ;
	const int netOutputSize = fcout_actsize * EXECUTOR_MAX_BATCH_SZ;

	XBLASHandle *handle = NULL;
	string kernelName = "kernelSxdnn_0";
	int ret = xblasCreate(handle, xclbin.c_str(), kernelName.c_str());
	assert(ret == 0);

	int depth = 3;
	int img_h[2], img_w[2];
	int batch_size = batch_sz;
	fprintf(stderr, "[CXDNN] Running image classification .........\n");
	float *input=(float*)malloc(height*width*depth*batch_size*sizeof(float));
	float *output=(float*)malloc(fpgaOutputSize*sizeof(float));
	float *inputPtr = input;
	float *outputPtr = output;

	int layercnt = 0;
	// load XDNN weights & create Executor
	XDNNScriptExecutor<float> *executor = loadExecutor(
			handle, dataDir, netCfgFile, quantCfgFile, layercnt);
	int unused;
	std::string unusedStr;
	std::vector<float> fcWeight, fcBias;
	ret = readFromFile(dataDir+"/fc_"+std::to_string(layercnt), unusedStr,
			unused, unused, unused, unused, fcWeight, false);
	assert(ret == 0);
	ret = readFromFile(dataDir+"/fc_bias_"+std::to_string(layercnt), unusedStr,
			unused, unused, unused, unused, fcBias, false);
	assert(ret == 0);
	// run on FPGA
	struct timeval start, end;
	DIR *dir_;
        const char * dir_path = dir_path_s.c_str();
        double total_latency=0.0,total_hw=0.0,total_sw=0.0;
	int status = 0;
        if(dir_path_s.length()!=0){
                dir_ = opendir(dir_path);
                if (dir_ == NULL) {
                        std::cout<<"Failed to open the input images folder"<<std::endl;
                        exit(0);
                }
                struct dirent *ent;
                int img_cnt=0;
                std::string thum="Thumbs.db";
                while ((ent = readdir(dir_)) != NULL) {
		if (ent->d_type == DT_REG) {
                                
				std::vector<std::string> img_path_t;
                                std::string image_file_path;

                                if(thum.compare(ent->d_name)!=0){
                                        image_file_path = dir_path_s + "/" + ent->d_name;
                                        img_path_t.push_back(image_file_path);
                                }
                                if(batch_size==2)
                                {
                                        ent=readdir(dir_);
                                        if(ent!=NULL){
                                                if(ent->d_type == DT_REG){
                                                        if(thum.compare(ent->d_name)!=0){
                                                                image_file_path = dir_path_s + "/" + ent->d_name;
                                                                img_path_t.push_back(image_file_path);
                                                        }
                                                }
                                        }
                                }
                                if(img_path_t.size()==batch_sz){

                                        std::vector<cv::Mat> in_frame;
                                        cv::Mat in_frame_t;
                                        for(int bi = 0; bi < batch_size; bi++)
                                        {
                                        //in_frame[bi] = cv::imread(img_path[bi],1);
                                                in_frame_t = cv::imread(img_path_t[bi],1);
                                                if(!in_frame_t.data)
                                                {
                                                        std :: cout << "[ERROR] Image read failed - " << img_path_t[bi] << std :: endl;
                                                        return -1;
                                                }
                                                else
                                                {
                                                        in_frame.push_back(in_frame_t);
                                                        //std :: cout << "[IMRDx] Image read : " << img_path_t[bi] << std :: endl;
	                                        }
				
						float *ptr = &input[bi*(height*width*depth)];
						//status = prepareInputData(height, width, depth, ptr, img_path[bi], img_h[bi], img_w[bi]);
						status = prepareInputData(in_frame[bi],height, width, depth, ptr,img_h, img_w);
						//if(status == 0)
						//	std::cout << "[CXDNN] prepareInputData done" << std::endl;

						gettimeofday(&start, 0);
						executor->execute(inputPtr, outputPtr, EXECUTOR_MAX_BATCH_SZ, 1);
						gettimeofday(&end, 0);
						total_hw=total_hw + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);

						//std::cout << PFX << "Profile CONV : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
						//std::cout << std::endl;
					
					}
					// get labels
					std::vector<std::string> labels = getLabels(labelFile);

					// run FC & Softmax on CPU
					gettimeofday(&start, 0);
					for(int bi = 0; bi < batch_size; bi++)
					{	
						// run FC
						std::vector<float> netOutput(netOutputSize);
						float *netoutputPtr = &(netOutput[0]);
						float *in_ptr = &outputPtr[bi*fpgaout_actsize];
						float *out_ptr = &netoutputPtr[bi*fcout_actsize];		
						computeFC(&(fcWeight[0]), &(fcBias[0]), in_ptr, 1, fcout_actsize, fpgaout_actsize, out_ptr);
						vector<float> sx_ptr(out_ptr, out_ptr+fcout_actsize);
						softmax(sx_ptr);

						// print top-1
						auto maxIt = std::max_element(sx_ptr.begin(), sx_ptr.end());
						int maxIdx = maxIt - sx_ptr.begin();
						std::cout << PFX << "Prediction: " << labels[maxIdx] << " ";
						std::cout << sx_ptr[maxIdx] << std::endl;
					}
					gettimeofday(&end, 0);

					total_sw=total_sw + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);
					//std::cout << PFX << "Profile sw layers : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
					//std::cout << std::endl;
				}
			}
			img_cnt++;
		}
		total_latency = total_sw + total_hw;
		std::cout << PFX << "Profile total sw layers: " <<total_sw/img_cnt<< " Profile total CONV: "<<total_hw/img_cnt<<" Profile total: " <<total_latency/img_cnt<< std::endl;
		std::cout << std::endl;
	}else {

		std::vector<cv::Mat> in_frame;
                cv::Mat in_frame_t;
                for(int bi = 0; bi < batch_size; bi++)
               	{
                        in_frame_t = cv::imread(img_path[bi],1);
                        if(!in_frame_t.data)
                        {
        			std :: cout << "[ERROR] Image read failed - " << img_path[bi] << std :: endl;
                                return -1;
                        }
                        else
                        {
                        	in_frame.push_back(in_frame_t);
                                //std :: cout << "[IMRDx] Image read : " << img_path_t[bi] << std :: endl;
                        }

                        float *ptr = &input[bi*(height*width*depth)];
                        //status = prepareInputData(height, width, depth, ptr, img_path[bi], img_h[bi], img_w[bi]);
                      	gettimeofday(&start, 0);
			status = prepareInputData(in_frame[bi],height, width, depth, ptr,img_h, img_w);
                      	gettimeofday(&end, 0);
                        //if(status == 0)
                        //std::cout << "[CXDNN] prepareInputData done" << std::endl;

                       std::cout << "\n\n[CXDNN] Profile input pre process : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
                       std::cout << std::endl;


                      gettimeofday(&start, 0);
                      executor->execute(inputPtr, outputPtr, EXECUTOR_MAX_BATCH_SZ, 1);
                      gettimeofday(&end, 0);

                      std::cout << PFX << "Profile CONV : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
                      std::cout << std::endl;

                }
                // get labels
                std::vector<std::string> labels = getLabels(labelFile);
		// run FC & Softmax on CPU
                gettimeofday(&start, 0);
                for(int bi = 0; bi < batch_size; bi++)
                {
                	// run FC
	        	std::vector<float> netOutput(netOutputSize);
        	      	float *netoutputPtr = &(netOutput[0]);
                      	float *in_ptr = &outputPtr[bi*fpgaout_actsize];
                        float *out_ptr = &netoutputPtr[bi*fcout_actsize];
                        computeFC(&(fcWeight[0]), &(fcBias[0]), in_ptr, 1, fcout_actsize, fpgaout_actsize, out_ptr);
                        vector<float> sx_ptr(out_ptr, out_ptr+fcout_actsize);
                        softmax(sx_ptr);

                        // print top-1
                        auto maxIt = std::max_element(sx_ptr.begin(), sx_ptr.end());
                        int maxIdx = maxIt - sx_ptr.begin();
                        std::cout << PFX << "Prediction: " << labels[maxIdx] << " ";
                        std::cout << sx_ptr[maxIdx] << std::endl;
         	}
                gettimeofday(&end, 0);
	        std::cout << PFX << "Profile sw layers : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
               	std::cout << std::endl;
	}

	// cleanup
	//xFree(*handle, cFpgaPtr);
	xblasDestroy(handle);
	free(input);
	free(output);

	return 0;
}
