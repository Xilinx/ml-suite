# DeepDetect Webcam Tutorial

# Introduction
This tutorial extends the [DeepDetect REST Tutorial][], using the [Deep Detect][] application and REST APIs to connect a webcam from a host machine to stream video to F1 and classify the images live. The full code of this project is in `/xfdnn_testdrive/deepdetect/`

For launching and connecting to instances, [start here][].

Start by launching Two Terminals

**Terminal 1**
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
4. Execute `./run.sh` to start the DeepDetect Caffe REST Server

	```
	$ ./start_deepdetect_docker.sh
	# sudo ./run.sh
	DeepDetect [ commit  ]

	INFO - 16:03:43 - Running DeepDetect HTTP server on 0.0.0.0:8080

	```

	When you see the message "INFO - 16:03:43 - Running DeepDetect HTTP server on 			0.0.0.0:8080" the scripted has started the webserver correctly.


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

4. Navigate to `/demo/webcam`
5. Edit `index.html` (using text editor such as nano)
	Find the section below:
	```html
	/*******************************************
	* TODO: Please update the address below
	* in 'url' to point to your public IP
	*******************************************/
	var url = "yourpublicdns.compute-1.amazonaws.com:8888";
	```
	Change yourpublicdns.compute-1.amazonaws.com with your instance public IP address.


   This is for client browser to upload image to server.py
6. Navigate to `deepdetect/`
7. Execute ./exec_deepdetect_docker.sh
8. Execute `python server.py`
   This starts the webcam demo webpage & server.
   ```
	$ cd demo/webcam/
	$ ls
	css  index.html  jpeg_camera  media  README  server.py  streams  test.html
	$ nano index.html
	$ cd ../..
	$ ./exec_deepdetect_docker.sh
	# cd demo/webcam/
	# python server.py
	serving at port 8888
	```

Once you see "serving at port 8888" the application is running and ready.

**Host PC:**
1. In Firefox visit http://yourpublicdns.compute-1.amazonaws.com:8888
	Firefox browsers work the best for this.
	Allow permissions for use of the Webcam and Adobe, if needed.

	![](img/deepdetect_allow.png)

	![](img/deepdetect_allow_a.png)

	Now you should be able to classify images via your webcam

	![](img/deepdetect_webcam.png)


[start here]: launching_instance.md
[DeepDetect REST Tutorial]: deepdetect_rest.md
[Deep Detect]: https://github.com/beniz/deepdetect
