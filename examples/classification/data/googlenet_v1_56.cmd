# template <id> XNMaxPool <kernel_w> <kernel_h>  <strides_w> <strides_h> <paddings_w> <paddings_h>  <inaddr> <insize_w> <insize_h> <inchan> <outaddr> <outsize_w> <outsize_h> <Bypass_Perf_Opt>
# template <id> XNAvgPool <kernel_w> <kernel_h>  <strides_w> <strides_h> <paddings_w> <paddings_h> <fcmode> <inaddr> <insize_w> <insize_h> <inchan> <outaddr> <outsize_w> <outsize_h> <Bypass_Perf_Opt>
# template <id> XNConv <layername> <kernel_w> <kernel_h> <strides_w> <strides_h> <padding_w> <padding_h> <dilation_w> <dilation_h> <preshift> <scale> <postshift> <relu> <bias> <inaddr> <insize_w> <insize_h> <inchan> <outaddr> <outsize_w> <outsize_h>  <outchan> <Bypass_Perf_Opt> 
# template <id> XNGather  <uram_dest> <ddr_src> <input_w> <input_h> <input_chan> 0 1 1 <start_row> <end_row>
# template <id> XNUpload <inaddr> <insize> <inchan>
# template <id> XNEltWise <name> <add> <bn> <relu> <inaddrA> <inaddrB> <insize_w> <insize_h> <inchan> <outaddr> <Bypass_Perf_Opt>
# template <id> XNScatter <uram_src> <ddr_dest> <input_w> <input_h> <input_chan> 0 1 1 <start_row> <end_row>
# # BUD data [u'Input'] ['layer'] data: type=Input, sizes=None, shapes=None, sched 0 Kernel None Strides None Padding None  NO VALID CODE  
2 XNConv conv1/7x7_s2 7 2 16 26 2 1 1 0x0 224 3 0x70000 112 64 0
# BUD "conv1/relu_7x7" [u'ReLU'] ['inplace_layer']
4 XNMaxPool pool1/3x3_s2 3 2 0 0x70000 112 64 0x0 56 0
5 XNConv conv2/3x3_reduce 1 1 16 26 2 1 1 0x0 56 64 0x70000 56 64 0
# BUD "conv2/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
7 XNConv conv2/3x3 3 1 16 26 2 1 1 0x70000 56 64 0xe0000 56 192 0
# BUD "conv2/relu_3x3" [u'ReLU'] ['inplace_layer']
9 XNMaxPool pool2/3x3_s2 3 2 0 0xe0000 56 192 0x0 28 0
10 XNConv inception_3a/1x1 1 1 16 26 2 1 1 0x0 28 192 0x420000 28 64 0
# BUD "inception_3a/relu_1x1" [u'ReLU'] ['inplace_layer']
12 XNConv inception_3a/3x3_reduce 1 1 16 26 2 1 1 0x0 28 192 0xa8000 28 96 0
# BUD "inception_3a/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
14 XNConv inception_3a/3x3 3 1 16 26 2 1 1 0xa8000 28 96 0x458000 28 128 0
# BUD "inception_3a/relu_3x3" [u'ReLU'] ['inplace_layer']
16 XNConv inception_3a/5x5_reduce 1 1 16 26 2 1 1 0x0 28 192 0xa8000 28 16 0
# BUD "inception_3a/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
18 XNConv inception_3a/5x5 5 1 16 26 2 1 1 0xa8000 28 16 0x4c8000 28 32 0
# BUD "inception_3a/relu_5x5" [u'ReLU'] ['inplace_layer']
20 XNMaxPool inception_3a/pool 3 1 1 0x0 28 192 0xa8000 28 0
21 XNConv inception_3a/pool_proj 1 1 16 26 2 1 1 0xa8000 28 192 0x4e4000 28 32 0
# BUD "inception_3a/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 23 XNConcat inception_3a/output  0x420000 0x500000 917504 
24 XNConv inception_3b/1x1 1 1 16 26 2 1 1 0x420000 28 256 0x27c000 28 128 0
# BUD "inception_3b/relu_1x1" [u'ReLU'] ['inplace_layer']
26 XNConv inception_3b/3x3_reduce 1 1 16 26 2 1 1 0x420000 28 256 0x0 28 128 0
# BUD "inception_3b/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
28 XNConv inception_3b/3x3 3 1 16 26 2 1 1 0x0 28 128 0x2ec000 28 192 0
# BUD "inception_3b/relu_3x3" [u'ReLU'] ['inplace_layer']
30 XNConv inception_3b/5x5_reduce 1 1 16 26 2 1 1 0x420000 28 256 0x0 28 32 0
# BUD "inception_3b/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
32 XNConv inception_3b/5x5 5 1 16 26 2 1 1 0x0 28 32 0x394000 28 96 0
# BUD "inception_3b/relu_5x5" [u'ReLU'] ['inplace_layer']
34 XNMaxPool inception_3b/pool 3 1 1 0x420000 28 256 0x0 28 0
35 XNConv inception_3b/pool_proj 1 1 16 26 2 1 1 0x0 28 256 0x3e8000 28 64 0
# BUD "inception_3b/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 37 XNConcat inception_3b/output  0x27c000 0x420000 1720320 
38 XNMaxPool pool3/3x3_s2 3 2 0 0x27c000 28 480 0x420000 14 0
39 XNConv inception_4a/1x1 1 1 16 26 2 1 1 0x420000 14 480 0x340000 14 192 0
# BUD "inception_4a/relu_1x1" [u'ReLU'] ['inplace_layer']
41 XNConv inception_4a/3x3_reduce 1 1 16 26 2 1 1 0x420000 14 480 0x0 14 96 0
# BUD "inception_4a/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
43 XNConv inception_4a/3x3 3 1 16 26 2 1 1 0x0 14 96 0x394000 14 208 0
# BUD "inception_4a/relu_3x3" [u'ReLU'] ['inplace_layer']
45 XNConv inception_4a/5x5_reduce 1 1 16 26 2 1 1 0x420000 14 480 0x4f2000 14 16 0
# BUD "inception_4a/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
47 XNConv inception_4a/5x5 5 1 16 26 2 1 1 0x4f2000 14 16 0x3ef000 14 48 0
# BUD "inception_4a/relu_5x5" [u'ReLU'] ['inplace_layer']
49 XNMaxPool inception_4a/pool 3 1 1 0x420000 14 480 0x0 14 0
50 XNConv inception_4a/pool_proj 1 1 16 26 2 1 1 0x0 14 480 0x404000 14 64 0
# BUD "inception_4a/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 52 XNConcat inception_4a/output  0x340000 0x420000 917504 
53 XNConv inception_4b/1x1 1 1 16 26 2 1 1 0x340000 14 512 0x420000 14 160 0
# BUD "inception_4b/relu_1x1" [u'ReLU'] ['inplace_layer']
55 XNConv inception_4b/3x3_reduce 1 1 16 26 2 1 1 0x340000 14 512 0x0 14 112 0
# BUD "inception_4b/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
57 XNConv inception_4b/3x3 3 1 16 26 2 1 1 0x0 14 112 0x466000 14 224 0
# BUD "inception_4b/relu_3x3" [u'ReLU'] ['inplace_layer']
59 XNConv inception_4b/5x5_reduce 1 1 16 26 2 1 1 0x340000 14 512 0x0 14 24 0
# BUD "inception_4b/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
61 XNConv inception_4b/5x5 5 1 16 26 2 1 1 0x0 14 24 0x4c8000 14 64 0
# BUD "inception_4b/relu_5x5" [u'ReLU'] ['inplace_layer']
63 XNMaxPool inception_4b/pool 3 1 1 0x340000 14 512 0x0 14 0
64 XNConv inception_4b/pool_proj 1 1 16 26 2 1 1 0x0 14 512 0x4e4000 14 64 0
# BUD "inception_4b/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 66 XNConcat inception_4b/output  0x420000 0x500000 917504 
67 XNConv inception_4c/1x1 1 1 16 26 2 1 1 0x420000 14 512 0x340000 14 128 0
# BUD "inception_4c/relu_1x1" [u'ReLU'] ['inplace_layer']
69 XNConv inception_4c/3x3_reduce 1 1 16 26 2 1 1 0x420000 14 512 0x0 14 128 0
# BUD "inception_4c/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
71 XNConv inception_4c/3x3 3 1 16 26 2 1 1 0x0 14 128 0x378000 14 256 0
# BUD "inception_4c/relu_3x3" [u'ReLU'] ['inplace_layer']
73 XNConv inception_4c/5x5_reduce 1 1 16 26 2 1 1 0x420000 14 512 0x0 14 24 0
# BUD "inception_4c/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
75 XNConv inception_4c/5x5 5 1 16 26 2 1 1 0x0 14 24 0x3e8000 14 64 0
# BUD "inception_4c/relu_5x5" [u'ReLU'] ['inplace_layer']
77 XNMaxPool inception_4c/pool 3 1 1 0x420000 14 512 0x0 14 0
78 XNConv inception_4c/pool_proj 1 1 16 26 2 1 1 0x0 14 512 0x404000 14 64 0
# BUD "inception_4c/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 80 XNConcat inception_4c/output  0x340000 0x420000 917504 
81 XNConv inception_4d/1x1 1 1 16 26 2 1 1 0x340000 14 512 0x259000 14 112 0
# BUD "inception_4d/relu_1x1" [u'ReLU'] ['inplace_layer']
83 XNConv inception_4d/3x3_reduce 1 1 16 26 2 1 1 0x340000 14 512 0x420000 14 144 0
# BUD "inception_4d/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
85 XNConv inception_4d/3x3 3 1 16 26 2 1 1 0x420000 14 144 0x28a000 14 288 0
# BUD "inception_4d/relu_3x3" [u'ReLU'] ['inplace_layer']
87 XNConv inception_4d/5x5_reduce 1 1 16 26 2 1 1 0x340000 14 512 0x420000 14 32 0
# BUD "inception_4d/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
89 XNConv inception_4d/5x5 5 1 16 26 2 1 1 0x420000 14 32 0x308000 14 64 0
# BUD "inception_4d/relu_5x5" [u'ReLU'] ['inplace_layer']
91 XNMaxPool inception_4d/pool 3 1 1 0x340000 14 512 0x420000 14 0
92 XNConv inception_4d/pool_proj 1 1 16 26 2 1 1 0x420000 14 512 0x324000 14 64 0
# BUD "inception_4d/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 94 XNConcat inception_4d/output  0x259000 0x340000 946176 
95 XNConv inception_4e/1x1 1 1 16 26 2 1 1 0x259000 14 528 0x394000 14 256 0
# BUD "inception_4e/relu_1x1" [u'ReLU'] ['inplace_layer']
97 XNConv inception_4e/3x3_reduce 1 1 16 26 2 1 1 0x259000 14 528 0x340000 14 160 0
# BUD "inception_4e/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
99 XNConv inception_4e/3x3 3 1 16 26 2 1 1 0x340000 14 160 0x404000 14 320 0
# BUD "inception_4e/relu_3x3" [u'ReLU'] ['inplace_layer']
101 XNConv inception_4e/5x5_reduce 1 1 16 26 2 1 1 0x259000 14 528 0x340000 14 32 0
# BUD "inception_4e/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
103 XNConv inception_4e/5x5 5 1 16 26 2 1 1 0x340000 14 32 0x490000 14 128 0
# BUD "inception_4e/relu_5x5" [u'ReLU'] ['inplace_layer']
105 XNMaxPool inception_4e/pool 3 1 1 0x259000 14 528 0x0 14 0
106 XNConv inception_4e/pool_proj 1 1 16 26 2 1 1 0x0 14 528 0x4c8000 14 128 0
# BUD "inception_4e/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 108 XNConcat inception_4e/output  0x394000 0x500000 1490944 
109 XNMaxPool pool4/3x3_s2 3 2 0 0x394000 14 832 0x0 7 0
110 XNConv inception_5a/1x1 1 1 16 26 2 1 1 0x0 7 832 0x44a000 7 256 0
# BUD "inception_5a/relu_1x1" [u'ReLU'] ['inplace_layer']
112 XNConv inception_5a/3x3_reduce 1 1 16 26 2 1 1 0x0 7 832 0xb6000 7 160 0
# BUD "inception_5a/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
114 XNConv inception_5a/3x3 3 1 16 26 2 1 1 0xb6000 7 160 0x482000 7 320 0
# BUD "inception_5a/relu_3x3" [u'ReLU'] ['inplace_layer']
116 XNConv inception_5a/5x5_reduce 1 1 16 26 2 1 1 0x0 7 832 0xb6000 7 32 0
# BUD "inception_5a/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
118 XNConv inception_5a/5x5 5 1 16 26 2 1 1 0xb6000 7 32 0x4c8000 7 128 0
# BUD "inception_5a/relu_5x5" [u'ReLU'] ['inplace_layer']
120 XNMaxPool inception_5a/pool 3 1 1 0x0 7 832 0xb6000 7 0
121 XNConv inception_5a/pool_proj 1 1 16 26 2 1 1 0xb6000 7 832 0x4e4000 7 128 0
# BUD "inception_5a/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 123 XNConcat inception_5a/output  0x44a000 0x500000 745472 
124 XNConv inception_5b/1x1 1 1 16 26 2 1 1 0x44a000 7 832 0x36a000 7 384 0
# BUD "inception_5b/relu_1x1" [u'ReLU'] ['inplace_layer']
126 XNConv inception_5b/3x3_reduce 1 1 16 26 2 1 1 0x44a000 7 832 0x0 7 192 0
# BUD "inception_5b/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
128 XNConv inception_5b/3x3 3 1 16 26 2 1 1 0x0 7 192 0x3be000 7 384 0
# BUD "inception_5b/relu_3x3" [u'ReLU'] ['inplace_layer']
130 XNConv inception_5b/5x5_reduce 1 1 16 26 2 1 1 0x44a000 7 832 0x0 7 48 0
# BUD "inception_5b/relu_5x5_reduce" [u'ReLU'] ['inplace_layer']
132 XNConv inception_5b/5x5 5 1 16 26 2 1 1 0x0 7 48 0x412000 7 128 0
# BUD "inception_5b/relu_5x5" [u'ReLU'] ['inplace_layer']
134 XNMaxPool inception_5b/pool 3 1 1 0x44a000 7 832 0x0 7 0
135 XNConv inception_5b/pool_proj 1 1 16 26 2 1 1 0x0 7 832 0x42e000 7 128 0
# BUD "inception_5b/relu_pool_proj" [u'ReLU'] ['inplace_layer']
# 137 XNConcat inception_5b/output  0x36a000 0x44a000 917504 
138 XNAvgPool pool5/7x7_s1 7 1 0 1 0x36a000 7 1024 0x44a000 1 0
# BUD "pool5/drop_7x7_s1" [u'Dropout'] ['inplace_layer']
# ## 140 XNInner loss3/classifier 16 26 2 0x44a000 1 1024 0x46a000 1000 1000 loss3/classifier: type=InnerProduct, sizes=None, shapes=[[1000, 1024], [1000]], sched 139 Kernel None Strides None Padding None  NO VALID CODE  
# # BUD prob [u'Softmax'] ['layer'] prob: type=Softmax, sizes=None, shapes=None, sched 140 Kernel None Strides None Padding None  NO VALID CODE  
