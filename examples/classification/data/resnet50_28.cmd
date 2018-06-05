1 XNConv conv1 7 2 16 26 2 1 1 0x1c0000 224 3 0x0 112 64
# BUD bn_conv1 [u'BatchNorm'] ['inplace_layer']
# BUD scale_conv1 [u'Scale'] ['inplace_layer']
# BUD conv1_relu [u'ReLU'] ['inplace_layer']
5 XNMaxPool pool1 3 2 0 0x0 112 64 0x1c0000 56 
5 XNUpload 0x1c0000 56 64
6 XNConv res2a_branch1 1 1 16 26 2 0 1 0x1c0000 56 64 0x240000 56 256
# BUD bn2a_branch1 [u'BatchNorm'] ['inplace_layer']
# BUD scale2a_branch1 [u'Scale'] ['inplace_layer']
9 XNConv res2a_branch2a 1 1 16 26 2 1 1 0x1c0000 56 64 0x0 56 64
# BUD bn2a_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale2a_branch2a [u'Scale'] ['inplace_layer']
# BUD res2a_branch2a_relu [u'ReLU'] ['inplace_layer']
13 XNConv res2a_branch2b 3 1 16 26 2 1 1 0x0 56 64 0x1d0000 56 64
# BUD bn2a_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale2a_branch2b [u'Scale'] ['inplace_layer']
# BUD res2a_branch2b_relu [u'ReLU'] ['inplace_layer']
17 XNConv res2a_branch2c 1 1 16 26 2 0 1 0x1d0000 56 64 0x10000 56 256
# BUD bn2a_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale2a_branch2c [u'Scale'] ['inplace_layer']
20 XNEltwise res2a 1 0 1 0x240000 0x10000 56 256 0x240000
20 XNUpload 0x240000 56 256
# BUD res2a_relu [u'ReLU'] ['inplace_layer']
22 XNConv res2b_branch2a 1 1 16 26 2 1 1 0x240000 56 256 0x0 56 64
# BUD bn2b_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale2b_branch2a [u'Scale'] ['inplace_layer']
# BUD res2b_branch2a_relu [u'ReLU'] ['inplace_layer']
26 XNConv res2b_branch2b 3 1 16 26 2 1 1 0x0 56 64 0x1d0000 56 64
# BUD bn2b_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale2b_branch2b [u'Scale'] ['inplace_layer']
# BUD res2b_branch2b_relu [u'ReLU'] ['inplace_layer']
30 XNConv res2b_branch2c 1 1 16 26 2 0 1 0x1d0000 56 64 0x10000 56 256
# BUD bn2b_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale2b_branch2c [u'Scale'] ['inplace_layer']
33 XNEltwise res2b 1 0 1 0x240000 0x10000 56 256 0x240000
33 XNUpload 0x240000 56 256
# BUD res2b_relu [u'ReLU'] ['inplace_layer']
35 XNConv res2c_branch2a 1 1 16 26 2 1 1 0x240000 56 256 0x0 56 64
# BUD bn2c_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale2c_branch2a [u'Scale'] ['inplace_layer']
# BUD res2c_branch2a_relu [u'ReLU'] ['inplace_layer']
39 XNConv res2c_branch2b 3 1 16 26 2 1 1 0x0 56 64 0x1d0000 56 64
# BUD bn2c_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale2c_branch2b [u'Scale'] ['inplace_layer']
# BUD res2c_branch2b_relu [u'ReLU'] ['inplace_layer']
43 XNConv res2c_branch2c 1 1 16 26 2 0 1 0x1d0000 56 64 0x10000 56 256
# BUD bn2c_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale2c_branch2c [u'Scale'] ['inplace_layer']
46 XNEltwise res2c 1 0 1 0x240000 0x10000 56 256 0x240000
46 XNUpload 0x240000 56 256
# BUD res2c_relu [u'ReLU'] ['inplace_layer']
48 XNConv res3a_branch1 1 2 16 26 2 0 1 0x240000 56 256 0x0 28 512
# BUD bn3a_branch1 [u'BatchNorm'] ['inplace_layer']
# BUD scale3a_branch1 [u'Scale'] ['inplace_layer']
51 XNConv res3a_branch2a 1 2 16 26 2 1 1 0x240000 56 256 0xe0000 28 128
# BUD bn3a_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale3a_branch2a [u'Scale'] ['inplace_layer']
# BUD res3a_branch2a_relu [u'ReLU'] ['inplace_layer']
55 XNConv res3a_branch2b 3 1 16 26 2 1 1 0xe0000 28 128 0x118000 28 128
# BUD bn3a_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale3a_branch2b [u'Scale'] ['inplace_layer']
# BUD res3a_branch2b_relu [u'ReLU'] ['inplace_layer']
59 XNConv res3a_branch2c 1 1 16 26 2 0 1 0x118000 28 128 0x150000 28 512
# BUD bn3a_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale3a_branch2c [u'Scale'] ['inplace_layer']
62 XNEltwise res3a 1 0 1 0x0 0x150000 28 512 0x0
62 XNUpload 0x0 28 512
# BUD res3a_relu [u'ReLU'] ['inplace_layer']
64 XNConv res3b_branch2a 1 1 16 26 2 1 1 0x0 28 512 0xe0000 28 128
# BUD bn3b_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale3b_branch2a [u'Scale'] ['inplace_layer']
# BUD res3b_branch2a_relu [u'ReLU'] ['inplace_layer']
68 XNConv res3b_branch2b 3 1 16 26 2 1 1 0xe0000 28 128 0x118000 28 128
# BUD bn3b_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale3b_branch2b [u'Scale'] ['inplace_layer']
# BUD res3b_branch2b_relu [u'ReLU'] ['inplace_layer']
72 XNConv res3b_branch2c 1 1 16 26 2 0 1 0x118000 28 128 0x150000 28 512
# BUD bn3b_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale3b_branch2c [u'Scale'] ['inplace_layer']
75 XNEltwise res3b 1 0 1 0x0 0x150000 28 512 0x0
75 XNUpload 0x0 28 512
# BUD res3b_relu [u'ReLU'] ['inplace_layer']
77 XNConv res3c_branch2a 1 1 16 26 2 1 1 0x0 28 512 0xe0000 28 128
# BUD bn3c_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale3c_branch2a [u'Scale'] ['inplace_layer']
# BUD res3c_branch2a_relu [u'ReLU'] ['inplace_layer']
81 XNConv res3c_branch2b 3 1 16 26 2 1 1 0xe0000 28 128 0x118000 28 128
# BUD bn3c_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale3c_branch2b [u'Scale'] ['inplace_layer']
# BUD res3c_branch2b_relu [u'ReLU'] ['inplace_layer']
85 XNConv res3c_branch2c 1 1 16 26 2 0 1 0x118000 28 128 0x150000 28 512
# BUD bn3c_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale3c_branch2c [u'Scale'] ['inplace_layer']
88 XNEltwise res3c 1 0 1 0x0 0x150000 28 512 0x0
88 XNUpload 0x0 28 512
# BUD res3c_relu [u'ReLU'] ['inplace_layer']
90 XNConv res3d_branch2a 1 1 16 26 2 1 1 0x0 28 512 0xe0000 28 128
# BUD bn3d_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale3d_branch2a [u'Scale'] ['inplace_layer']
# BUD res3d_branch2a_relu [u'ReLU'] ['inplace_layer']
94 XNConv res3d_branch2b 3 1 16 26 2 1 1 0xe0000 28 128 0x118000 28 128
# BUD bn3d_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale3d_branch2b [u'Scale'] ['inplace_layer']
# BUD res3d_branch2b_relu [u'ReLU'] ['inplace_layer']
98 XNConv res3d_branch2c 1 1 16 26 2 0 1 0x118000 28 128 0x150000 28 512
# BUD bn3d_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale3d_branch2c [u'Scale'] ['inplace_layer']
101 XNEltwise res3d 1 0 1 0x0 0x150000 28 512 0x0
101 XNUpload 0x0 28 512
# BUD res3d_relu [u'ReLU'] ['inplace_layer']
103 XNConv res4a_branch1 1 2 16 26 2 0 1 0x0 28 512 0xe0000 14 1024
# BUD bn4a_branch1 [u'BatchNorm'] ['inplace_layer']
# BUD scale4a_branch1 [u'Scale'] ['inplace_layer']
106 XNConv res4a_branch2a 1 2 16 26 2 1 1 0x0 28 512 0x1c0000 14 256
# BUD bn4a_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4a_branch2a [u'Scale'] ['inplace_layer']
# BUD res4a_branch2a_relu [u'ReLU'] ['inplace_layer']
110 XNConv res4a_branch2b 3 1 16 26 2 1 1 0x1c0000 14 256 0x0 14 256
# BUD bn4a_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4a_branch2b [u'Scale'] ['inplace_layer']
# BUD res4a_branch2b_relu [u'ReLU'] ['inplace_layer']
114 XNConv res4a_branch2c 1 1 16 26 2 0 1 0x0 14 256 0x1c0000 14 1024
# BUD bn4a_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4a_branch2c [u'Scale'] ['inplace_layer']
117 XNEltwise res4a 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
117 XNUpload 0xe0000 14 1024
# BUD res4a_relu [u'ReLU'] ['inplace_layer']
119 XNConv res4b_branch2a 1 1 16 26 2 1 1 0xe0000 14 1024 0x0 14 256
# BUD bn4b_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4b_branch2a [u'Scale'] ['inplace_layer']
# BUD res4b_branch2a_relu [u'ReLU'] ['inplace_layer']
123 XNConv res4b_branch2b 3 1 16 26 2 1 1 0x0 14 256 0x38000 14 256
# BUD bn4b_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4b_branch2b [u'Scale'] ['inplace_layer']
# BUD res4b_branch2b_relu [u'ReLU'] ['inplace_layer']
127 XNConv res4b_branch2c 1 1 16 26 2 0 1 0x38000 14 256 0x1c0000 14 1024
# BUD bn4b_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4b_branch2c [u'Scale'] ['inplace_layer']
130 XNEltwise res4b 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
130 XNUpload 0xe0000 14 1024
# BUD res4b_relu [u'ReLU'] ['inplace_layer']
132 XNConv res4c_branch2a 1 1 16 26 2 1 1 0xe0000 14 1024 0x0 14 256
# BUD bn4c_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4c_branch2a [u'Scale'] ['inplace_layer']
# BUD res4c_branch2a_relu [u'ReLU'] ['inplace_layer']
136 XNConv res4c_branch2b 3 1 16 26 2 1 1 0x0 14 256 0x38000 14 256
# BUD bn4c_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4c_branch2b [u'Scale'] ['inplace_layer']
# BUD res4c_branch2b_relu [u'ReLU'] ['inplace_layer']
140 XNConv res4c_branch2c 1 1 16 26 2 0 1 0x38000 14 256 0x1c0000 14 1024
# BUD bn4c_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4c_branch2c [u'Scale'] ['inplace_layer']
143 XNEltwise res4c 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
143 XNUpload 0xe0000 14 1024
# BUD res4c_relu [u'ReLU'] ['inplace_layer']
145 XNConv res4d_branch2a 1 1 16 26 2 1 1 0xe0000 14 1024 0x0 14 256
# BUD bn4d_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4d_branch2a [u'Scale'] ['inplace_layer']
# BUD res4d_branch2a_relu [u'ReLU'] ['inplace_layer']
149 XNConv res4d_branch2b 3 1 16 26 2 1 1 0x0 14 256 0x38000 14 256
# BUD bn4d_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4d_branch2b [u'Scale'] ['inplace_layer']
# BUD res4d_branch2b_relu [u'ReLU'] ['inplace_layer']
153 XNConv res4d_branch2c 1 1 16 26 2 0 1 0x38000 14 256 0x1c0000 14 1024
# BUD bn4d_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4d_branch2c [u'Scale'] ['inplace_layer']
156 XNEltwise res4d 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
156 XNUpload 0xe0000 14 1024
# BUD res4d_relu [u'ReLU'] ['inplace_layer']
158 XNConv res4e_branch2a 1 1 16 26 2 1 1 0xe0000 14 1024 0x0 14 256
# BUD bn4e_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4e_branch2a [u'Scale'] ['inplace_layer']
# BUD res4e_branch2a_relu [u'ReLU'] ['inplace_layer']
162 XNConv res4e_branch2b 3 1 16 26 2 1 1 0x0 14 256 0x38000 14 256
# BUD bn4e_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4e_branch2b [u'Scale'] ['inplace_layer']
# BUD res4e_branch2b_relu [u'ReLU'] ['inplace_layer']
166 XNConv res4e_branch2c 1 1 16 26 2 0 1 0x38000 14 256 0x1c0000 14 1024
# BUD bn4e_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4e_branch2c [u'Scale'] ['inplace_layer']
169 XNEltwise res4e 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
169 XNUpload 0xe0000 14 1024
# BUD res4e_relu [u'ReLU'] ['inplace_layer']
171 XNConv res4f_branch2a 1 1 16 26 2 1 1 0xe0000 14 1024 0x0 14 256
# BUD bn4f_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale4f_branch2a [u'Scale'] ['inplace_layer']
# BUD res4f_branch2a_relu [u'ReLU'] ['inplace_layer']
175 XNConv res4f_branch2b 3 1 16 26 2 1 1 0x0 14 256 0x38000 14 256
# BUD bn4f_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale4f_branch2b [u'Scale'] ['inplace_layer']
# BUD res4f_branch2b_relu [u'ReLU'] ['inplace_layer']
179 XNConv res4f_branch2c 1 1 16 26 2 0 1 0x38000 14 256 0x1c0000 14 1024
# BUD bn4f_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale4f_branch2c [u'Scale'] ['inplace_layer']
182 XNEltwise res4f 1 0 1 0xe0000 0x1c0000 14 1024 0xe0000
182 XNUpload 0xe0000 14 1024
# BUD res4f_relu [u'ReLU'] ['inplace_layer']
184 XNConv res5a_branch1 1 2 16 26 2 0 1 0xe0000 14 1024 0x0 7 2048
# BUD bn5a_branch1 [u'BatchNorm'] ['inplace_layer']
# BUD scale5a_branch1 [u'Scale'] ['inplace_layer']
187 XNConv res5a_branch2a 1 2 16 26 2 1 1 0xe0000 14 1024 0x1c0000 7 512
# BUD bn5a_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale5a_branch2a [u'Scale'] ['inplace_layer']
# BUD res5a_branch2a_relu [u'ReLU'] ['inplace_layer']
191 XNConv res5a_branch2b 3 1 16 26 2 1 1 0x1c0000 7 512 0xe0000 7 512
# BUD bn5a_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale5a_branch2b [u'Scale'] ['inplace_layer']
# BUD res5a_branch2b_relu [u'ReLU'] ['inplace_layer']
195 XNConv res5a_branch2c 1 1 16 26 2 0 1 0xe0000 7 512 0x118000 7 2048
# BUD bn5a_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale5a_branch2c [u'Scale'] ['inplace_layer']
198 XNEltwise res5a 1 0 1 0x0 0x118000 7 2048 0x0
198 XNUpload 0x0 7 2048
# BUD res5a_relu [u'ReLU'] ['inplace_layer']
200 XNConv res5b_branch2a 1 1 16 26 2 1 1 0x0 7 2048 0xe0000 7 512
# BUD bn5b_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale5b_branch2a [u'Scale'] ['inplace_layer']
# BUD res5b_branch2a_relu [u'ReLU'] ['inplace_layer']
204 XNConv res5b_branch2b 3 1 16 26 2 1 1 0xe0000 7 512 0x118000 7 512
# BUD bn5b_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale5b_branch2b [u'Scale'] ['inplace_layer']
# BUD res5b_branch2b_relu [u'ReLU'] ['inplace_layer']
208 XNConv res5b_branch2c 1 1 16 26 2 0 1 0x118000 7 512 0x150000 7 2048
# BUD bn5b_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale5b_branch2c [u'Scale'] ['inplace_layer']
211 XNEltwise res5b 1 0 1 0x0 0x150000 7 2048 0x0
211 XNUpload 0x0 7 2048
# BUD res5b_relu [u'ReLU'] ['inplace_layer']
213 XNConv res5c_branch2a 1 1 16 26 2 1 1 0x0 7 2048 0xe0000 7 512
# BUD bn5c_branch2a [u'BatchNorm'] ['inplace_layer']
# BUD scale5c_branch2a [u'Scale'] ['inplace_layer']
# BUD res5c_branch2a_relu [u'ReLU'] ['inplace_layer']
217 XNConv res5c_branch2b 3 1 16 26 2 1 1 0xe0000 7 512 0x118000 7 512
# BUD bn5c_branch2b [u'BatchNorm'] ['inplace_layer']
# BUD scale5c_branch2b [u'Scale'] ['inplace_layer']
# BUD res5c_branch2b_relu [u'ReLU'] ['inplace_layer']
221 XNConv res5c_branch2c 1 1 16 26 2 0 1 0x118000 7 512 0x150000 7 2048
# BUD bn5c_branch2c [u'BatchNorm'] ['inplace_layer']
# BUD scale5c_branch2c [u'Scale'] ['inplace_layer']
224 XNEltwise res5c 1 0 1 0x0 0x150000 7 2048 0x0
224 XNUpload 0x0 7 2048
# BUD res5c_relu [u'ReLU'] ['inplace_layer']
226 XNAvgPool pool5 7 1 0 1 0x0 7 2048 0xe0000 1
226 XNUpload 0xe0000 1 2048
# ## 227 XNInner fc1000 16 26 2 0xe0000 1 2048 0x0 1000 1000 fc1000: type=InnerProduct, sizes=None, shapes=[[1000, 2048], [1000]], sched 226 Kernel None Strides None Padding None  NO VALID CODE  
# # BUD prob [u'Softmax'] ['layer'] prob: type=Softmax, sizes=None, shapes=None, sched 227 Kernel None Strides None Padding None  NO VALID CODE  
