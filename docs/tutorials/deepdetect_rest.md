# DeepDetect REST Tutorial

## Introduction
This tutorial describes how to launch [Deep Detect][], a deep learning API and web server application which has integrations for REST APIs and uses F1 for Image classification acceleration.

The source for this project is available in the Test Drive at: `/home/centos/xfdnn_18_04_02/deepdetect/`

**Note: This Tutorial is only available on [Amazon AWS EC2 F1][]**

For instructions on launching and connecting to aws instances, see [here][].

Start by launching Two Terminals.

**Terminal 1:**
1. Connect to F1
2. Navigate to `/home/centos/xfdnn_18_04_02/deepdetect/`
	```
	$ cd /home/centos/xfdnn_18_04_02/deepdetect/
	$ ls
	createService.sh           libs     sdaccel_profile_summary.csv   testService.sh
	dede                       models   sdaccel_profile_summary.html  xclbin
	demo                       run.sh   start_deepdetect_docker.sh    xfdnn_scheduler
	exec_deepdetect_docker.sh  runtime  templates
	```
2. Execute `./start_deepdetect_docker.sh` to enter application docker
3. Navigate to `/opt/deepdetect/`
4. Execute `./runDeepDetectServer.sh` to start the DeepDetect Caffe REST Server
	```
	$ ./start_deepdetect_docker.sh
	# ./runDeepDetectServer.sh
	DeepDetect [ commit  ]

	INFO - 16:03:43 - Running DeepDetect HTTP server on 0.0.0.0:8080
	```

	When you see the message "INFO - 16:03:43 - Running DeepDetect HTTP server on 0.0.0.0:8080", this indicates that the script has started the webserver correctly.</br>When the FPGA is ready you will see `XBLAS online! (d=0)`

**Terminal 2:**
1. Connect to F1
2. Navigate to `/home/centos/xfdnn_18_04_02/deepdetect/`
	```
	$ cd /home/centos/xfdnn_18_04_02/deepdetect/
	$ ls
	createService.sh           libs     sdaccel_profile_summary.csv   testService.sh
	dede                       models   sdaccel_profile_summary.html  xclbin
	demo                       run.sh   start_deepdetect_docker.sh    xfdnn_scheduler
	exec_deepdetect_docker.sh  runtime  templates
	```
3. Execute `./createService.sh`
   This initializes the DeepDetect server in Terminal 1 with GoogLeNet-v1. </br>
   Wait for the FPGA to load xclbin in Terminal 1. </br>
   On success you will see `{"status":{"code":201,"msg":"Created"}}`

   More service can be added for the following networks:
 - Flowers-102 : createServicesFlowers.sh
 - Places-365  : createServicePlaces.sh
 - Resnet-50   : createServiceResnet.h

4. To verify your service is working, execute `./test.sh`
	```
	$ ./createService.sh
	{"status":{"code":201,"msg":"Created"}}
	$ ./testService.sh
	  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
									 Dload  Upload   Total   Spent    Left  Speed
	100   589  100   398  100   191    405    194 --:--:-- --:--:-- --:--:--   405
	{
		"body": {
			"predictions": [
				{
					"classes": [
						{
							"cat": "n02088364 beagle",
							"prob": 0.8565296530723572
						},
						{
							"cat": "n02089867 Walker hound, Walker foxhound",
							"prob": 0.09222473204135895
						},
						{
							"cat": "",
							"last": true,
							"prob": 0.090826615691185
						}
					],
					"uri": "https://www.dogbreedinfo.com/images24/BeagleBayleePurebredDogs8Months1.jpg"
				}
			]
		},
		"head": {
			"method": "/predict",
			"service": "imageserv",
			"time": 980.0
		},
		"status": {
			"code": 200,
			"msg": "OK"
		}
	}
	```

5. Navigate to `demo/imgdetect`
	```
	$ cd demo/imgdetect/
	$ ./run.sh
	```

**Host PC:**

This starts the web server from where you can submit URLs.

1. Visit `http://<yourpublicdns>.compute-1.amazonaws.com` from your broswer

	![](img/deepdetect_rest.png)

[here]: docs/tutorials/launching_instance.md
[Deep Detect]: https://github.com/beniz/deepdetect
