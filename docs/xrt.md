# Installing XRT for Xilinx ML Suite
Currently ML Suite is dependent on Xilinx XRT 2018.2

1. Download the XRT package for your OS
```
## Choose from the below ##
# Ubuntu 16.04

wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_201802.2.1.127_16.04-xrt.deb && \
 sudo apt install ./*.deb

# Ubuntu 18.04
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_201802.2.1.127_18.04-xrt.deb && \
 sudo apt install ./*.deb

# CentOS/RHEL
wget https://www.xilinx.com/bin/public/openDownload?filename=xrt_201802.2.1.127_7.4.1708-xrt.rpm && \
 sudo yum install ./*.rpm
```

2. Download the shell firmware for your Xilinx platform  
This step must be done in a browser and you must use your Xilinx login. After downloading install using the corresponding apt/yum install command.  
  
| Hardware Platform | Operating System | Link |  
|-------------------|------------------|------|  
| vcu1525 | Ubuntu 16.04 | [VCU1525 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vcu1525-dynamic-16.04.deb) |
| vcu1525 | Ubuntu 18.04 | [VCU1525 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vcu1525-dynamic-18.04.deb) |
| vcu1525 | RHEL/CentOS | [[VCU1525 .rpm](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vcu1525-dynamic-5.1-2342198.x86_64.rpm) |
| alveo-u200 | Ubuntu 16.04 | [U200 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u200-xdma-16.04.deb) |
| alveo-u200 | Ubuntu 18.04 | [U200 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u200-xdma-18.04.deb) |
| alveo-u200 | RHEL/CentOS | [U200 .rpm](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u200-xdma-201820.1-2342198.x86_64.rpm) |
| alveo-u250 | Ubuntu 16.04 | [U250 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u250-xdma-16.04.deb) |
| alveo-u250 | Ubuntu 18.04 | [U250 .deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u250-xdma-18.04.deb) |
| alveo-u250 | RHEL/CentOS | [U250 .rpm](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-u250-xdma-201820.1-2342198.x86_64.rpm) |

3. After installing the shell firmware to the host system, you will be prompted to use XRT to flash the firmware to the FPGA card. 
Follow the instruction, and after completion cold-reboot the system (Power off, Power On).

4. Verify installation
```
# The below command should be executable, and it should return details about your system
/opt/xilinx/xrt/bin/xbutil query
```
