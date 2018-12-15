# template id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h fcmode inaddr insize_w insize_h inchan outaddr outsize_w outsize_h Bypass_Perf_Opt 
# template id XNOp uram_dest ddr_src input_w input_h input_chan a0 b1 c1 start_row end_row 
# template id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt 
# template id XNOp name kernel_w kernel_h  strides_w strides_h paddings_w paddings_h  inaddr insize_w insize_h inchan outaddr outsize_w outsize_h Bypass_Perf_Opt 
# template id XNOp uram_src ddr_dest input_w input_h input_chan a0 b1 c1 start_row end_row 
# template id XNOp inaddr insize inchan
# template id XNOp name add bn relu inaddrA inaddrB insize_w insize_h inchan outaddr Bypass_Perf_Opt 
# template id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt 
# template id XNOp name kernel_w kernel_h strides_w strides_h padding_w padding_h dilation_w dilation_h preshift scale postshift relu bias inaddr insize_w insize_h inchan outaddr outsize_w outsize_h  outchan Bypass_Perf_Opt  pool_kernel_w pool_kernel_h pool_strides_w pool_strides_h pool_paddings_w pool_paddings_h pool_fcmode pool_inaddr pool_insize_w pool_insize_h pool_inchan pool_outaddr pool_outsize_w pool_outsize_h
# 1 Input download not possible
3 XNScatter 0x0 0x1a20000 608 608 3 0 1 1 0 607 # # data_blob 
4 XNGather 0x0 0x1a20000 608 608 3 0 1 1 0 101 # #  Conv conv0 # SPLIT Code :)
5 XNConv conv0#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 608 102 3 0x8c400 608 101 32 0 # SPLIT Code :)
6 XNScatter 0x8c400 0x0 608 608 32 0 1 1 0 100 # # Conv conv0 # SPLIT Code :)
7 XNGather 0x0 0x1a20000 608 608 3 0 1 1 100 202 # #  Conv conv0 # SPLIT Code :)
8 XNConv conv0#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 608 103 3 0x8da00 608 101 32 0 # SPLIT Code :)
9 XNScatter 0x8da00 0x0 608 608 32 0 1 1 101 201 # # Conv conv0 # SPLIT Code :)
10 XNGather 0x0 0x1a20000 608 608 3 0 1 1 201 303 # #  Conv conv0 # SPLIT Code :)
11 XNConv conv0#2 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 608 103 3 0x8da00 608 101 32 0 # SPLIT Code :)
12 XNScatter 0x8da00 0x0 608 608 32 0 1 1 202 302 # # Conv conv0 # SPLIT Code :)
13 XNGather 0x0 0x1a20000 608 608 3 0 1 1 302 404 # #  Conv conv0 # SPLIT Code :)
14 XNConv conv0#3 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 608 103 3 0x8da00 608 101 32 0 # SPLIT Code :)
15 XNScatter 0x8da00 0x0 608 608 32 0 1 1 303 403 # # Conv conv0 # SPLIT Code :)
16 XNGather 0x0 0x1a20000 608 608 3 0 1 1 403 505 # #  Conv conv0 # SPLIT Code :)
17 XNConv conv0#4 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 608 103 3 0x8da00 608 101 32 0 # SPLIT Code :)
18 XNScatter 0x8da00 0x0 608 608 32 0 1 1 404 504 # # Conv conv0 # SPLIT Code :)
19 XNGather 0x0 0x1a20000 608 608 3 0 1 1 504 607 # #  Conv conv0 # SPLIT Code :)
20 XNConv conv0#5 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 608 104 3 0x8f000 608 103 32 0 # SPLIT Code :)
21 XNScatter 0x8f000 0x0 608 608 32 0 1 1 505 607 # # Conv conv0 # SPLIT Code :)
24 XNGather 0x0 0x0 608 608 32 0 1 1 0 75 # #  Poolpool1 # SPLIT Code :)
25 XNMaxPool pool1#0 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
26 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 0 37 # #  Poolpool1 # SPLIT Code :)
27 XNGather 0x0 0x0 608 608 32 0 1 1 76 151 # #  Poolpool1 # SPLIT Code :)
28 XNMaxPool pool1#1 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
29 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 38 75 # #  Poolpool1 # SPLIT Code :)
30 XNGather 0x0 0x0 608 608 32 0 1 1 152 227 # #  Poolpool1 # SPLIT Code :)
31 XNMaxPool pool1#2 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
32 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 76 113 # #  Poolpool1 # SPLIT Code :)
33 XNGather 0x0 0x0 608 608 32 0 1 1 228 303 # #  Poolpool1 # SPLIT Code :)
34 XNMaxPool pool1#3 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
35 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 114 151 # #  Poolpool1 # SPLIT Code :)
36 XNGather 0x0 0x0 608 608 32 0 1 1 304 379 # #  Poolpool1 # SPLIT Code :)
37 XNMaxPool pool1#4 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
38 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 152 189 # #  Poolpool1 # SPLIT Code :)
39 XNGather 0x0 0x0 608 608 32 0 1 1 380 455 # #  Poolpool1 # SPLIT Code :)
40 XNMaxPool pool1#5 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
41 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 190 227 # #  Poolpool1 # SPLIT Code :)
42 XNGather 0x0 0x0 608 608 32 0 1 1 456 531 # #  Poolpool1 # SPLIT Code :)
43 XNMaxPool pool1#6 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
44 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 228 265 # #  Poolpool1 # SPLIT Code :)
45 XNGather 0x0 0x0 608 608 32 0 1 1 532 607 # #  Poolpool1 # SPLIT Code :)
46 XNMaxPool pool1#7 2 2 2 2 0 0 0x0 608 76 32 0x344000 304 38 0 # SPLIT Code :)
47 XNScatter 0x344000 0x1a20000 304 304 32 0 1 1 266 303 # #  Poolpool1 # SPLIT Code :)
50 XNGather 0x0 0x1a20000 304 304 32 0 1 1 0 60 # #  Conv conv2 # SPLIT Code :)
51 XNConv conv2#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 304 61 32 0x16e000 304 60 64 0 # SPLIT Code :)
52 XNScatter 0x16e000 0x0 304 304 64 0 1 1 0 59 # # Conv conv2 # SPLIT Code :)
53 XNGather 0x0 0x1a20000 304 304 32 0 1 1 59 120 # #  Conv conv2 # SPLIT Code :)
54 XNConv conv2#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 304 62 32 0x174000 304 60 64 0 # SPLIT Code :)
55 XNScatter 0x174000 0x0 304 304 64 0 1 1 60 119 # # Conv conv2 # SPLIT Code :)
56 XNGather 0x0 0x1a20000 304 304 32 0 1 1 119 180 # #  Conv conv2 # SPLIT Code :)
57 XNConv conv2#2 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 304 62 32 0x174000 304 60 64 0 # SPLIT Code :)
58 XNScatter 0x174000 0x0 304 304 64 0 1 1 120 179 # # Conv conv2 # SPLIT Code :)
59 XNGather 0x0 0x1a20000 304 304 32 0 1 1 179 240 # #  Conv conv2 # SPLIT Code :)
60 XNConv conv2#3 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 304 62 32 0x174000 304 60 64 0 # SPLIT Code :)
61 XNScatter 0x174000 0x0 304 304 64 0 1 1 180 239 # # Conv conv2 # SPLIT Code :)
62 XNGather 0x0 0x1a20000 304 304 32 0 1 1 239 303 # #  Conv conv2 # SPLIT Code :)
63 XNConv conv2#4 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 304 65 32 0x186000 304 64 64 0 # SPLIT Code :)
64 XNScatter 0x186000 0x0 304 304 64 0 1 1 240 303 # # Conv conv2 # SPLIT Code :)
67 XNGather 0x0 0x0 304 304 64 0 1 1 0 75 # #  Poolpool3 # SPLIT Code :)
68 XNMaxPool pool3#0 2 2 2 2 0 0 0x0 304 76 64 0x390000 152 38 0 # SPLIT Code :)
69 XNScatter 0x390000 0xe40000 152 152 64 0 1 1 0 37 # #  Poolpool3 # SPLIT Code :)
70 XNGather 0x0 0x0 304 304 64 0 1 1 76 151 # #  Poolpool3 # SPLIT Code :)
71 XNMaxPool pool3#1 2 2 2 2 0 0 0x0 304 76 64 0x390000 152 38 0 # SPLIT Code :)
72 XNScatter 0x390000 0xe40000 152 152 64 0 1 1 38 75 # #  Poolpool3 # SPLIT Code :)
73 XNGather 0x0 0x0 304 304 64 0 1 1 152 227 # #  Poolpool3 # SPLIT Code :)
74 XNMaxPool pool3#2 2 2 2 2 0 0 0x0 304 76 64 0x390000 152 38 0 # SPLIT Code :)
75 XNScatter 0x390000 0xe40000 152 152 64 0 1 1 76 113 # #  Poolpool3 # SPLIT Code :)
76 XNGather 0x0 0x0 304 304 64 0 1 1 228 303 # #  Poolpool3 # SPLIT Code :)
77 XNMaxPool pool3#3 2 2 2 2 0 0 0x0 304 76 64 0x390000 152 38 0 # SPLIT Code :)
78 XNScatter 0x390000 0xe40000 152 152 64 0 1 1 114 151 # #  Poolpool3 # SPLIT Code :)
81 XNGather 0x0 0xe40000 152 152 64 0 1 1 0 50 # #  Conv conv4 # SPLIT Code :)
82 XNConv conv4#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 152 51 64 0x132000 152 50 128 0 # SPLIT Code :)
83 XNScatter 0x132000 0x0 152 152 128 0 1 1 0 49 # # Conv conv4 # SPLIT Code :)
84 XNGather 0x0 0xe40000 152 152 64 0 1 1 49 100 # #  Conv conv4 # SPLIT Code :)
85 XNConv conv4#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 152 52 64 0x138000 152 50 128 0 # SPLIT Code :)
86 XNScatter 0x138000 0x0 152 152 128 0 1 1 50 99 # # Conv conv4 # SPLIT Code :)
87 XNGather 0x0 0xe40000 152 152 64 0 1 1 99 151 # #  Conv conv4 # SPLIT Code :)
88 XNConv conv4#2 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 152 53 64 0x13e000 152 52 128 0 # SPLIT Code :)
89 XNScatter 0x13e000 0x0 152 152 128 0 1 1 100 151 # # Conv conv4 # SPLIT Code :)
92 XNGather 0x0 0x0 152 152 128 0 1 1 0 49 # #  Conv conv5 # SPLIT Code :)
93 XNConv conv5#0 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 152 50 128 0x258000 152 50 64 0 # SPLIT Code :)
94 XNScatter 0x258000 0x720000 152 152 64 0 1 1 0 49 # # Conv conv5 # SPLIT Code :)
95 XNGather 0x0 0x0 152 152 128 0 1 1 50 99 # #  Conv conv5 # SPLIT Code :)
96 XNConv conv5#1 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 152 50 128 0x258000 152 50 64 0 # SPLIT Code :)
97 XNScatter 0x258000 0x720000 152 152 64 0 1 1 50 99 # # Conv conv5 # SPLIT Code :)
98 XNGather 0x0 0x0 152 152 128 0 1 1 100 151 # #  Conv conv5 # SPLIT Code :)
99 XNConv conv5#2 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 152 52 128 0x270000 152 52 64 0 # SPLIT Code :)
100 XNScatter 0x270000 0x720000 152 152 64 0 1 1 100 151 # # Conv conv5 # SPLIT Code :)
103 XNGather 0x0 0x720000 152 152 64 0 1 1 0 50 # #  Conv conv6 # SPLIT Code :)
104 XNConv conv6#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 152 51 64 0x132000 152 50 128 0 # SPLIT Code :)
105 XNScatter 0x132000 0x0 152 152 128 0 1 1 0 49 # # Conv conv6 # SPLIT Code :)
106 XNGather 0x0 0x720000 152 152 64 0 1 1 49 100 # #  Conv conv6 # SPLIT Code :)
107 XNConv conv6#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 152 52 64 0x138000 152 50 128 0 # SPLIT Code :)
108 XNScatter 0x138000 0x0 152 152 128 0 1 1 50 99 # # Conv conv6 # SPLIT Code :)
109 XNGather 0x0 0x720000 152 152 64 0 1 1 99 151 # #  Conv conv6 # SPLIT Code :)
110 XNConv conv6#2 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 152 53 64 0x13e000 152 52 128 0 # SPLIT Code :)
111 XNScatter 0x13e000 0x0 152 152 128 0 1 1 100 151 # # Conv conv6 # SPLIT Code :)
114 XNGather 0x0 0x0 152 152 128 0 1 1 0 75 # #  Poolpool7 # SPLIT Code :)
115 XNMaxPool pool7#0 2 2 2 2 0 0 0x0 152 76 128 0x390000 76 38 0 # SPLIT Code :)
116 XNScatter 0x390000 0x720000 76 76 128 0 1 1 0 37 # #  Poolpool7 # SPLIT Code :)
117 XNGather 0x0 0x0 152 152 128 0 1 1 76 151 # #  Poolpool7 # SPLIT Code :)
118 XNMaxPool pool7#1 2 2 2 2 0 0 0x0 152 76 128 0x390000 76 38 0 # SPLIT Code :)
119 XNScatter 0x390000 0x720000 76 76 128 0 1 1 38 75 # #  Poolpool7 # SPLIT Code :)
122 XNGather 0x0 0x720000 76 76 128 0 1 1 0 38 # #  Conv conv8 # SPLIT Code :)
123 XNConv conv8#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 76 39 128 0x138000 76 38 256 0 # SPLIT Code :)
124 XNScatter 0x138000 0x0 76 76 256 0 1 1 0 37 # # Conv conv8 # SPLIT Code :)
125 XNGather 0x0 0x720000 76 76 128 0 1 1 37 75 # #  Conv conv8 # SPLIT Code :)
126 XNConv conv8#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 76 39 128 0x138000 76 38 256 0 # SPLIT Code :)
127 XNScatter 0x138000 0x0 76 76 256 0 1 1 38 75 # # Conv conv8 # SPLIT Code :)
130 XNGather 0x0 0x0 76 76 256 0 1 1 0 37 # #  Conv conv9 # SPLIT Code :)
131 XNConv conv9#0 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 76 38 256 0x260000 76 38 128 0 # SPLIT Code :)
132 XNScatter 0x260000 0x4c0000 76 76 128 0 1 1 0 37 # # Conv conv9 # SPLIT Code :)
133 XNGather 0x0 0x0 76 76 256 0 1 1 38 75 # #  Conv conv9 # SPLIT Code :)
134 XNConv conv9#1 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 76 38 256 0x260000 76 38 128 0 # SPLIT Code :)
135 XNScatter 0x260000 0x4c0000 76 76 128 0 1 1 38 75 # # Conv conv9 # SPLIT Code :)
138 XNGather 0x0 0x4c0000 76 76 128 0 1 1 0 38 # #  Conv conv10 # SPLIT Code :)
139 XNConv conv10#0 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 76 39 128 0x138000 76 38 256 0 # SPLIT Code :)
140 XNScatter 0x138000 0x0 76 76 256 0 1 1 0 37 # # Conv conv10 # SPLIT Code :)
141 XNGather 0x0 0x4c0000 76 76 128 0 1 1 37 75 # #  Conv conv10 # SPLIT Code :)
142 XNConv conv10#1 3 3 1 1 1 0 1 1 16 26 2 1 1 0x0 76 39 128 0x138000 76 38 256 0 # SPLIT Code :)
143 XNScatter 0x138000 0x0 76 76 256 0 1 1 38 75 # # Conv conv10 # SPLIT Code :)
146 XNGather 0x0 0x0 76 76 256 0 1 1 0 37 # #  Poolpool11 # SPLIT Code :)
147 XNMaxPool pool11#0 2 2 2 2 0 0 0x0 76 38 256 0x260000 38 19 0 # SPLIT Code :)
148 XNScatter 0x260000 0x4c0000 38 38 256 0 1 1 0 18 # #  Poolpool11 # SPLIT Code :)
149 XNGather 0x0 0x0 76 76 256 0 1 1 38 75 # #  Poolpool11 # SPLIT Code :)
150 XNMaxPool pool11#1 2 2 2 2 0 0 0x0 76 38 256 0x260000 38 19 0 # SPLIT Code :)
151 XNScatter 0x260000 0x4c0000 38 38 256 0 1 1 19 37 # #  Poolpool11 # SPLIT Code :)
154 XNGather 0x0 0x4c0000 38 38 256 0 1 1 0 37 # # pool11_blob 
155 XNConv conv12 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 38 38 256 0x130000 38 38 512 0
157 XNConv conv13 1 1 1 1 0 0 1 1 16 26 2 1 1 0x130000 38 38 512 0x0 38 38 256 0
159 XNConv conv14 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 38 38 256 0x130000 38 38 512 0
161 XNConv conv15 1 1 1 1 0 0 1 1 16 26 2 1 1 0x130000 38 38 512 0x0 38 38 256 0
163 XNConv conv16 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 38 38 256 0x130000 38 38 512 0
165 XNMaxPool pool17 2 2 2 2 0 0 0x130000 38 38 512 0x0 19 19 0
167 XNScatter 0x130000 0x0 38 38 512 0 1 1 0 37 # # conv16_blob 
168 XNConv conv18 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 19 19 512 0x130000 19 19 1024 0
170 XNConv conv19 1 1 1 1 0 0 1 1 16 26 2 1 1 0x130000 19 19 1024 0x0 19 19 512 0
172 XNConv conv20 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 19 19 512 0x130000 19 19 1024 0
174 XNConv conv21 1 1 1 1 0 0 1 1 16 26 2 1 1 0x130000 19 19 1024 0x0 19 19 512 0
176 XNConv conv22 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 19 19 512 0x130000 19 19 1024 0
178 XNScatter 0x130000 0x260000 19 19 1024 0 1 1 0 18 # # conv22_blob 
179 XNGather 0x260000 0x260000 19 19 1024 0 1 1 0 18 # # conv22_blob 
180 XNConv conv23 3 3 1 1 1 1 1 1 16 26 2 1 1 0x260000 19 19 1024 0x0 19 19 1024 0
182 XNConv conv24 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 19 19 1024 0x286000 19 19 1024 0
184 XNGather 0x0 0x0 38 38 512 0 1 1 0 37 # # conv16_blob 
185 XNScatter 0x260000 0x0 19 19 1088 0 1 1 0 18 # # concat28_blob 
186 XNConv conv26 1 1 1 1 0 0 1 1 16 26 2 1 1 0x0 38 38 512 0x260000 38 38 64 0
188 XNScatter 0x260000 0x143000 38 38 64 0 1 1 0 37 # # conv26_blob 
189 XNGather 0x0 0x0 19 19 1088 0 1 1 0 18 # # concat28_blob 
190 XNGather 0x286000 0x143000 38 38 64 0 1 1 0 37 # # conv26_blob 
191 XNMaxPool pool27 2 2 2 2 0 0 0x286000 38 38 64 0x0 19 19 0
# 193 XNConcat concat28  0x0 0x286000 2646016 
195 XNConv conv29 3 3 1 1 1 1 1 1 16 26 2 1 1 0x0 19 19 1088 0x286000 19 19 1024 0
197 XNConv conv30 1 1 1 1 0 0 1 1 16 26 2 0 1 0x286000 19 19 1024 0x0 19 19 425 0
