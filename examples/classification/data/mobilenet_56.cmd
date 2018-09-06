# template XNAvgPool id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h fcmode inaddr insize_w insize_h inchan outaddr outsize_w outsize_h Bypass_Perf_Opt 
# template XNGather id XNOp uram_dest ddr_src input_w input_h input_chan a0 b1 c1 start_row end_row 
# template XNDeconv id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt 
# template XNMaxPool id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h  inaddr insize_w insize_h inchan outaddr outsize_w outsize_h Bypass_Perf_Opt 
# template XNScatter id XNOp uram_src ddr_dest input_w input_h input_chan a0 b1 c1 start_row end_row 
# template XNUpload id XNOp inaddr insize inchan
# template XNEltwise id XNOp name add bn relu inaddrA inaddrB insize_w insize_h inchan outaddr Bypass_Perf_Opt 
# template XNConv id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt 
# template XNConvP id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt  pool_kernel_w pool_kernel_h pool_strides_w pool_strides_h pool_paddings_w pool_paddings_h pool_fcmode pool_inaddr pool_insize_w pool_insize_h pool_inchan pool_outaddr pool_outsize_w pool_outsize_h
# 1 Input download not possible
2 XNConv conv1 3 3 2 2 1 1 1 1 16 26 2 1 1 0x0 224 224 3 0x70000 112 112 32 0
3 XNConvDepth conv2_1/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x70000 112 112 32 0x150000 112 112 32 0
4 XNConv conv2_1/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0x150000 112 112 32 0x230000 112 112 64 0
5 XNConvDepth conv2_2/dw 3 3 2 2 1 1 1 1 16 26 2 1 1 0x230000 112 112 64 0x0 56 56 64 0
6 XNConv conv2_2/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 56 56 64 0x70000 56 56 128 0
7 XNConvDepth conv3_1/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x70000 56 56 128 0x150000 56 56 128 0
8 XNConv conv3_1/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0x150000 56 56 128 0x0 56 56 128 0
9 XNConvDepth conv3_2/dw 3 3 2 2 1 1 1 1 16 26 2 1 1 0x0 56 56 128 0xe0000 28 28 128 0
10 XNConv conv3_2/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 28 28 128 0x0 28 28 256 0
11 XNConvDepth conv4_1/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 28 28 256 0xe0000 28 28 256 0
12 XNConv conv4_1/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 28 28 256 0x0 28 28 256 0
13 XNConvDepth conv4_2/dw 3 3 2 2 1 1 1 1 16 26 2 1 1 0x0 28 28 256 0xe0000 14 14 256 0
14 XNConv conv4_2/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 256 0x0 14 14 512 0
15 XNConvDepth conv5_1/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 14 14 512 0
16 XNConv conv5_1/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 512 0x0 14 14 512 0
17 XNConvDepth conv5_2/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 14 14 512 0
18 XNConv conv5_2/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 512 0x0 14 14 512 0
19 XNConvDepth conv5_3/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 14 14 512 0
20 XNConv conv5_3/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 512 0x0 14 14 512 0
21 XNConvDepth conv5_4/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 14 14 512 0
22 XNConv conv5_4/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 512 0x0 14 14 512 0
23 XNConvDepth conv5_5/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 14 14 512 0
24 XNConv conv5_5/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 14 14 512 0x0 14 14 512 0
25 XNConvDepth conv5_6/dw 3 3 2 2 1 1 1 1 16 26 2 1 1 0x0 14 14 512 0xe0000 7 7 512 0
26 XNConv conv5_6/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 7 7 512 0x0 7 7 1024 0
27 XNConvDepth conv6/dw 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 7 7 1024 0xe0000 7 7 1024 0
28 XNConv conv6/sep 1 1 1 1 0 0 1 1 16 26 2 1 1 0xe0000 7 7 1024 0x0 7 7 1024 0
29 XNAvgPool pool6 1 1 1 1 0 0 1 0x0 7 7 1024 0xe0000 1 1 0
# ## 30 XNInner fc7 16 26 2 0xe0000 1 1024 0x0 1 1000 fc7: type=InnerProduct, sizes=None, shapes=SizeType(batches=1, channels=1, height=[1000, 1024], width=[1000]), sched 29 Kernel None Strides None Padding None  NO VALID CODE  
