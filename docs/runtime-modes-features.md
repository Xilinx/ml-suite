# Runtime modes and features

`TODO: index`

## Latency and Throughput modes

The XDNN processor can run in two modes, Latency and Throughput. In Latency mode, the compiler generates a schedule that runs on the Main Block of the processor. Since all operations are done by one block, this minimizes data movement overheads and delivers the lowest latency.

In Throughput mode, the compiler generates a schedule that utilizes a Dedicated Block in addition to the Main Block. The Dedicated Block is a smaller block that will only run the first convolution operation in the network. This is to exploit the property that the first convolution makes up most of the computation ops in most networks.

Both the Dedicated and Main Blocks are configured to work in a pipelined manner, trading latency for throughput. Specifically:
- The host application sends input activation to the Dedicated Block to execute the first Convolution
- The result from the Dedicated Block is streamed to DDR, and the the Main Block is signalled to start
- The Main Block streams the activation data from DDR and executes the remaining operations of the network
- The host reads the output of the Main Block

In Throughput mode, the throughput of the XDNN processor is determined by the maximum latency of the Main Block or Dedicated Block.

## Asynchronous execution

The runtime engine allows the user application to submit concurrent inference tasks to the XDNN processor asynchronously. This is so that:
- the user application can ensure that there is always work queued up to keep the FPGA 100% busy
- pre-processing and post-processing tasks before and after FPGA execution can run in parallel
- the user application is free to do other things and decide when to block and collect results from the FPGA

We use the following scheme for enqueuing execution asynchronously and collecting results:
```
  fpgaRT.exec_async()
  fpgaRT.get_result()
```

## Streaming mode

For maximum throughput, it is recommended to take advantage of the XDNN processor's ability to run in a streaming manner.  A typical inference flow may include the following steps: 
- A. Decode image
- B. Pre-process image
- C. Transfer image to FPGA
- D. Execute network on FPGA
- E. Read result from FPGA
- F. Post-process image
It is not necessary to wait for all steps to finish for an image before proceeding to the next image. As soon as image 1 is done with step A, image 2 can be submitted to step A while image 1 moves to step B.

## Single PE and multiple model

## Multiple PE and different models

## Multiple PE and same model

## Single model multiple PE

## Multiple FPGA execution
