//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
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
using namespace cv;
using namespace std;
#define PFX "[CXDNN] "
#define EXECUTOR_MAX_BATCH_SZ 2
// Note: this is assuming V1 format (without separate width/height)
int readFromFile(const string &fname, string &layerName, 
		int &kernWidth, int &kernHeight, int &inc, int &outc,
		std::vector<float> &vals, bool hasHeader=true)
{
	ifstream f(fname.c_str());
	if (!f.good())
		return -1;

	//std::cout << "[CXDNN] In readFromFile() : Bias Size : outc : " << outc << std::end;
	if (hasHeader)
	{
		f >> layerName;
		f >> kernWidth;
		f >> inc;
		f >> outc;
		//std::cout << "[CXDNN] In readFromFile() : Bias Size : outc : " << outc << std::end;
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
	XDNNScriptExecutor<float> *executor
	= new XDNNScriptExecutor<float>(handles, m, netCfgFile, quantCfgFile,
			/*scaleB*/30,/*cuMask*/ 0);
	return executor;
		}

std::vector<float> loadInput(const string &dataDir){
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
		}


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
int swlayers(int batch_size, float *fpgaOutput, bbox *bboxes,int in_height, int in_width,int *img_h, int *img_w, std::vector<std::string> labels,int out_w,int out_h,int out_d,std::vector<cv::Mat> &in_frame)
{
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
		int startidx = batch_id*outsize;
		softmaxout = fpgaOutput + startidx;

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
			fprintf(stderr, "\n\n[CXDNN] Obj %d: %s", j, labels[bboxes[j].classid].c_str());
			fprintf(stderr, "\n[CXDNN] Score = %f", bboxes[j].prob); 
			fprintf(stderr, "\n[CXDNN] (xlo,ylo) = (%d,%d)", bboxes[j].xlo,  bboxes[j].ylo);
			fprintf(stderr, "\n[CXDNN] (xhi,yhi) = (%d,%d)", bboxes[j].xhi,  bboxes[j].yhi);
			cv::rectangle(in_frame[batch_id], cv::Point(bboxes[j].xhi,bboxes[j].yhi),cv::Point(bboxes[j].xlo,bboxes[j].ylo),Scalar(255,0,0),2,8,0);
		}
		cv::imwrite("yolo_out.jpg",in_frame[batch_id]);
	} // batch_id

	return 0;
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
        cout<<" --xclbin=<MLsuite ROOT PATH>/overlaybins/1525/overlay_2.xclbin --data_dir=<MLsuite ROOT PATH>/apps/yolo/work/yolov2.caffemodel_data --cmd_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo.cmds.json --quant_json=<MLsuite ROOT PATH>/apps/yolo/work/yolo_deploy_608_8b.json --labelfile=<MLsuite ROOT PATH>/examples/classification/coco_names.txt --in_img/--in_img_dir=<MLsuite ROOT PATH>/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=608 --in_h=608 --out_w=19 --out_h=19 --out_d=425" <<endl;
}
void ProcessArgs(int argc, char** argv,std::map<std::string ,std::string> &arg_map,std::vector<std::string> &app_args,string &xclbin,string &dataDir,string &netCfgFile,string &quantCfgFile,string &labelFile,int &height,int &width,int &out_h,int &out_w,int &out_d,int &batch_sz,string &dir_path){

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

}

int main(int argc, char** argv)
{

	std::map<std::string ,std::string> arg_map;
        std::vector<std::string> app_args;
        std::vector<std::string>img_path;
        string xclbin,dataDir,netCfgFile,quantCfgFile,labelFile;
        int height,width,out_h,out_w,out_d,batch_sz;
	std::string dir_path_s;
	ProcessArgs(argc,argv,arg_map,app_args,xclbin,dataDir,netCfgFile,quantCfgFile,labelFile,height,width,out_h,out_w,out_d,batch_sz,dir_path_s);
	const int fpgaOutputSize = out_h*out_w*out_d * batch_sz; 
	XBLASHandle *handle = NULL;
	string kernelName = "kernelSxdnn_0";
	int ret = xblasCreate(handle, xclbin.c_str(), kernelName.c_str());
	assert(ret == 0);

	
	int depth = 3;
	int batch_size = batch_sz;//EXECUTOR_MAX_BATCH_SZ;
	int img_h, img_w;

	fprintf(stderr, "[CXDNN] Running yolo .........\n");
	float *input=(float*)malloc(height*width*depth*batch_size*sizeof(float));
	
	// run on FPGA
	struct timeval start, end;

	int status = 0;

	float *output=(float*)malloc(fpgaOutputSize*sizeof(float));// = loadInput(dataDir);
	float *inputPtr = input;
	float *outputPtr = output;

	// make FPGA pointer for output (inputs are auto-created)
	//XMemPtr *cFpgaPtr = xMalloc(*handle, fpgaOutputSize*sizeof(float));
	//xMemcpy(*handle, outputPtr, cFpgaPtr, fpgaOutputSize*sizeof(float));
	
	int layercnt = 0;
	// load XDNN weights & create Executor
	XDNNScriptExecutor<float> *executor = loadExecutor(
			handle, dataDir, netCfgFile, quantCfgFile, layercnt);

	DIR *dir_;
	const char * dir_path = dir_path_s.c_str();
	double total_latency=0.0;
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
						float *ptr = input;
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

						status = prepareInputData(in_frame[bi],height, width, depth, ptr+bi*height*width*depth,&img_h, &img_w);
					//if(status == 0)
					//	std::cout << "[CXDNN] prepareInputData done" << std::endl;
					}
				
				//gettimeofday(&end, 0);
				//gettimeofday(&start, 0);
	
				//std::cout << "\n\n[CXDNN] Profile input pre process : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
					gettimeofday(&start, 0);
					executor->execute(inputPtr, outputPtr,batch_sz,0);
					gettimeofday(&end, 0);
				//gettimeofday(&end, 0);
				//std::cerr<< "[CXDNN] Profile CONV : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
				//std::cerr << std::endl;
				// get labels
					std::vector<std::string> labels = getLabels(labelFile);
	
					bbox *bboxes;
				//gettimeofday(&start, 0);
					swlayers(batch_size, outputPtr, bboxes, height,width,&img_h, &img_w, labels,out_w,out_h,out_d,in_frame);
					total_latency = total_latency + (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec);	
					img_cnt++;
				}
			}
		}	
		std::cout << "\n\n[CXDNN] Profile total latency in usec : " << (total_latency)/img_cnt <<" / BATCH SIZE "<<" num of calls :"<<img_cnt<<std::endl;
	}
#if 0
	else{
		
		cv::Mat in_frame[EXECUTOR_MAX_BATCH_SZ];
                for(int bi = 0; bi < batch_size; bi++)
        	{
                	float *ptr = input;
			in_frame[bi] = cv::imread(img_path[bi],1);
                        if(!in_frame[bi].data)
                        {
                        	std :: cout << "[ERROR] Image read failed - " << img_path[bi] << std :: endl;
                                return -1;
                       	}
			else
                        {
                        	std :: cout << "[IMRDx] Image read : " << img_path[bi] << std :: endl;
                        }

                        status = prepareInputData(in_frame[bi],height, width, depth, ptr+bi*height*width*depth,&img_h, &img_w);
                        if(status == 0)
                        	std::cout << "[CXDNN] prepareInputData done" << std::endl;
                }
                        gettimeofday(&end, 0);
                        gettimeofday(&start, 0);

                        std::cout << "\n\n[CXDNN] Profile input pre process : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
                        executor->execute(inputPtr, outputPtr,EXECUTOR_MAX_BATCH_SZ,0);
                        gettimeofday(&end, 0);
                        std::cerr<< "[CXDNN] Profile CONV : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;
                        std::cerr << std::endl;
                        // get labels
                        std::vector<std::string> labels = getLabels(labelFile);
                        bbox *bboxes;
                        gettimeofday(&start, 0);
                        swlayers(batch_size, outputPtr, bboxes, height,width,&img_h, &img_w, labels,out_w,out_h,out_d,in_frame);
                        gettimeofday(&end, 0);

                       std::cout << "\n\n[CXDNN] Profile sw layers : " << (end.tv_sec * 1e6 + end.tv_usec) - (start.tv_sec * 1e6 + start.tv_usec) << std::endl;	
	}
#endif	
	// cleanup
	//xblasDestroy(handle);
	//free(bboxes);
	//free(input);
	//free(output);
	return 0;
}
