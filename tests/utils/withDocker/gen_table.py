import sys
import re

file_name = sys.argv[1]
#print file_name

with open(file_name, "r") as f:
    lines= f.readlines()

full_data = []
gen_info = {}

idx = 0
while(idx < len(lines)):
    line = lines[idx].strip()

    if not line:
        #print "not a line : ", line, idx
        idx += 1
        continue
#    else:
#        print line, idx
   
    if "ID" in line:
        kw = lines[idx].split(":")[-1]
        gen_info['git_id'] = kw

    if "Platform" in line:
        kw = lines[idx].split(":")[-1]
        gen_info['platform'] = kw

    if "*** Network :" in line:
        data = {}
        kw = line.split(":")[-1]
        data['network_name'] =  kw
        data["latency mode hw_counter in ms"] = "0"
        data['latency mode exec_xdnn in ms'] = "0"
        data['throughput mode hw_counter in ms'] = "0"
        data['throughput mode exec_xdnn in ms'] = "0"
        data["latency mode HW acc"] = "0"
        data["throughput mode HW acc"] = "0"
        data['compile_status'] = "Success"
        data['HW_status'] = "Success"
        data['compile_mode'] = "default"

        nw_start = True
        latency_mode = False
        read_top1 = False
        read_top5 = False
        idx += 1
        #print "loop 2 : ", line, idx
        while(nw_start and idx < len(lines)):
            #print "loop 3 : ", line, idx
            line = lines[idx].strip()
            if not line:
                idx += 1
                continue

            if "compile mode" in line:
                 kw = line.split(":")[-1]
                 data['compile_mode'] = kw
            
            if "hw_counter" in line and data['network_name'] == ' yolov2' :
                 #print lines[idx-1].strip()
		 reobj = re.findall(r'-?\d+\.\d+', line.strip())
                 l1 = reobj[0]
                 #reobj= re.match('\[XDNN\]   hw_counter     : (\d+\.\d+) ms', line.strip())
                 #l1 = reobj.groups()
                 latency = ''.join(l1)
                 data["latency mode hw_counter in ms"] = latency
            
            if "exec_xdnn" in line and data['network_name'] == ' yolov2':
                 reobj= re.match('\[XDNN\]   exec_xdnn      : (\d+\.\d+) ms', line.strip())
                 l1 = reobj.groups()
                 latency = ''.join(l1)
                 data["latency mode exec_xdnn in ms"] = latency
            

            if "hw_counter" in line and "Run mode : latency" in lines[idx-1].strip():
                 #print lines[idx-1].strip()
                 #print line
                 #reobj= re.match('\[XDNN\]   hw_counter     : (\d+\.\d+) ms', line.strip())
		 reobj = re.findall(r'-?\d+\.\d+', line.strip())
		 #print reobj
		 if reobj is not None:
                 	l1 = reobj[0]
                 	#l1 = reobj.groups()
                 	latency = ''.join(l1)
                 	data["latency mode hw_counter in ms"] = latency
           
            if "mAP:" in line and "Run mode : latency" in lines[idx-3].strip():
                 reobj= re.match('mAP: (\d+\.\d+)', line.strip())
                 mAP = reobj.groups()
                 score = ''.join(mAP)
                 data["latency mode HW acc"] = str(round(float(score)*100, 2))

            if "exec_xdnn" in line and "Run mode : latency" in lines[idx-2].strip():
                 reobj= re.match('\[XDNN\]   exec_xdnn      : (\d+\.\d+) ms', line.strip())
                 l1 = reobj.groups()
                 latency = ''.join(l1)
                 data["latency mode exec_xdnn in ms"] = latency
            
            if "mean average precision" in line and "Run mode : latency" in lines[idx-3].strip():
                 reobj= re.match('mean average precision \(mAP\) = (\d+\.\d+) (\d+\.\d+)', line.strip())
                 mavg, mavg_per = reobj.groups()
                 data['latency mode HW acc'] = mavg_per 


            if "Run mode : latency" in line:
                latency_mode = True
            
            if "Run mode : throughput" in line:
                latency_mode = False

            if "Average:" in line and "top-5" in line or "top5" in line and "xfdnn/subgraph0" not in line :
#                print line
                reobj = re.findall(r'\d+\.\d+', line.strip())
                top5 = reobj[1]
                read_top5 = True
            
            if "Average:" in line and "top-1" in line or "accuracy  --" in line and "xfdnn/subgraph0" not in line : 
                #print line
                reobj = re.findall(r'\d+\.\d+', line.strip())
                top1 = reobj[1]
                read_top1 = True

            if read_top1 and read_top5 :
                data["top1"] = round(float(top1)*100, 2)
                data["top5"] = round(float(top5)*100, 2)

                read_top1 = False
                read_top5 = False

                if latency_mode :
                    data["latency mode HW acc"] = str(data['top1']) + ' / ' + str(data['top5'])
                else :
                    data["throughput mode HW acc"] = str(data['top1']) + ' / ' + str(data['top5'])

            
            if "hw_counter" in line and "Run mode : throughput" in lines[idx-1].strip():
                 #print lines[idx-1].strip()
                 #print line
		 reobj = re.findall(r'-?\d+\.\d+', line.strip())
                 #reobj= re.match('\[XDNN\]   hw_counter     : (\d+\.\d+) ms', line.strip())
                 l1 = reobj[0]
                 #l1 = reobj.groups()
                 latency = ''.join(l1)
                 #print latency
                 #print type(latency)
                 data["throughput mode hw_counter in ms"] = latency
           

            if "subgraph0/latency" in line and "Run mode : throughput" in lines[idx-1].strip():
                 #print lines[idx-1].strip()
                 print line
                 reobj= re.match('xfdnn/subgraph0/latency  -- This Batch:  \[(\d+\.\d+)\]  Average:  \[(\d+\.\d+)]  Batch#:  (\d+)', line.strip())
                 cur_lat, avg_lat, batch_num = reobj.groups()
                 #reobj= re.match('\[XDNN\]   hw_counter     : (\d+\.\d+) ms', line.strip())
                 #l1 = reobj.groups()
                 #latency = ''.join(l1)
                 #print latency
                 #print type(latency)
                 data["throughput mode hw_counter in ms"] = avg_lat
           
            if "exec_xdnn" in line and "Run mode : throughput" in lines[idx-2].strip():
                 reobj= re.match('\[XDNN\]   exec_xdnn      : (\d+\.\d+) ms', line.strip())
                 l1 = reobj.groups()
                 latency = ''.join(l1)
                 data["throughput mode exec_xdnn in ms"] = latency
            
            if "mean average precision" in line and "Run mode : throughput" in lines[idx-3].strip():
                 reobj= re.match('mean average precision \(mAP\) = (\d+\.\d+) (\d+\.\d+)', line.strip())
                 mavg, mavg_per = reobj.groups()
                 data['throughput mode HW acc'] = mavg_per 
            if "mAP:" in line and "Run mode : throughput" in lines[idx-3].strip():
                 #print line
                 reobj= re.match('mAP: (\d+\.\d+)', line.strip())
                 mAP = reobj.groups()
                 score = ''.join(mAP)
                 data["throughput mode HW acc"] =  str(round(float(score)*100, 2))
            
            if "compiler error" in line:
                kw = line.split("-")[-1]
                data['compile_status'] = 'Error : ' + kw 
 
            if "ERROR" in line:
                kw = line.split(":")[-1]
                data['HW_status'] =  'Error : ' + kw 
            
            if "*** Network End" in line:
                nw_start = False

            idx += 1
        
        full_data.append(data)

    idx += 1

fields = ['network_name', 'compile_mode', "latency mode hw_counter in ms", 'latency mode exec_xdnn in ms','throughput mode hw_counter in ms', 'throughput mode exec_xdnn in ms', \
            'latency mode HW acc','throughput mode HW acc', 'compile_status', 'HW_status']

with open('xfdnn_nightly.csv', 'w') as f:
    f.write('Platform, ' + gen_info['platform'])
    f.write('Commit ID, ' + gen_info['git_id'] + '\n')
    f.write(', '.join(fields) + '\n')
    for item in full_data:
        vals = [item[key] for key in fields]
        f.write(', '.join(vals)+ '\n')


        

