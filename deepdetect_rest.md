# DeepDetect REST Tutorial

For launching and connecting to instances, [start here][].

Start by launching Two Terminals.

**Terminal 1:**
1. Connect to F1
2. Navigate to `/xfdnn_testdrive/deepdetect/`
	```
	$ cd xfdnn_testdrive/deepdetect/
	$ ls
	createService.sh           libs     sdaccel_profile_summary.csv   test.sh
	dede                       models   sdaccel_profile_summary.html  xclbin
	demo                       run.sh   start_deepdetect_docker.sh    xfdnn_scheduler
	exec_deepdetect_docker.sh  runtime  templates
	```
2. Execute `./start_deepdetect_docker.sh` to enter application docker
3. Navigate to `/deepdetect/`
4. Execute `sudo ./run.sh` to start the DeepDetect Caffe REST Server
	```
	$ ./start_deepdetect_docker.sh
	# sudo ./run.sh
	DeepDetect [ commit  ]

	INFO - 16:03:43 - Running DeepDetect HTTP server on 0.0.0.0:8080
	```

	When you see the message "INFO - 16:03:43 - Running DeepDetect HTTP server on 0.0.0.0:8080" the scripted has started the webserver correctly.

**Terminal 2:**
1. Connect to F1
2. Navigate to `/xfdnn_testdrive/deepdetect/`
	```
	$ cd xfdnn_testdrive/deepdetect/
	$ ls
	createService.sh           libs     sdaccel_profile_summary.csv   test.sh
	dede                       models   sdaccel_profile_summary.html  xclbin
	demo                       run.sh   start_deepdetect_docker.sh    xfdnn_scheduler
	exec_deepdetect_docker.sh  runtime  templates
	```
3. Execute `./createService.sh`
   This initializes the DeepDetect server in Terminal 1. </br>
   Wait for FPGA to load xclbin in Terminal 1. </br>
   On success you will see `{"status":{"code":201,"msg":"Created"}}`

4. To verify your service is working, execute `./test.sh`
	```
	$ ./createService.sh
	{"status":{"code":201,"msg":"Created"}}
	$ ./test.sh
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
	This starts the web server where you can submit URLs.
6. Visit `http://yourpublicdns.compute-1.amazonaws.com` from your broswer

	![](img/deepdetect_rest.png)

[start here]: launching_instance.md
