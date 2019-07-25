// Copyright (c) 2017
// Xilinx, Inc.
// All rights reserved.
// 
// No part of this document may be copied, transmitted or
// disclosed in any form or fashion without the express
// written consent of Xilinx, Inc.

/**
 *  @brief Xilinx DNN library for FPGA acceleration
 *
 *  @author Aaron Ng (aaronn@xilinx.com)
 */

#ifndef XDNN_H
#define XDNN_H
#include <future>
#include "xdnn_env.h"
#include "xmlrt.h"
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>


class XDNNLayerQuantParam {
  public:
    std::string name;
    std::vector<int> preShift;
    std::vector<int> scale;
    std::vector<int> posScale;
    std::vector<int> negScale;
    std::vector<int> postShift;
    std::vector<float> thresh;
    int bitWidth;
    float threshIn;
    float threshOut;
};

class XDNNLayerQuantParamDB {
  public:
    static XDNNLayerQuantParam *getLayerQuantParam(
        const std::string &cfgFile, const std::string &layerName);
  private:
    XDNNLayerQuantParamDB();
};


class XDNNPacketExecutor : public XPacketExecutor {
public:
  virtual ~XDNNPacketExecutor() {}
  void prepBForDevice(XPipelinePacket &packet) override;
  void prepCForDevice(XPipelinePacket &packet) override;
  void unpackCFromDevice(XPipelinePacket &packet) override;
  void execute(XPipelinePacket &packet) const override;
  XPacketExecutorGetCompletedFunc getCompletedFunc() override;
};

std::string getUniqueLayerName(std::string layerName);
bool ddrFormatNeeded(int, int);
bool is_power_of_2(int);
template <typename Dtype>
struct XDNNWeightParam
{
  const Dtype * wgt_;
  const Dtype * bias_;
  size_t wgt_sz_;
  size_t bias_sz_;
};

class XPipelinePacket;
class XDNNDescriptor;
class XDNNPoolDescriptor;
class XDNNConvolutionDescriptor;
class XDNNDataDescriptor;
class XDNNAvgpoolDescriptor;
class XDNNMaxpoolDescriptor;
class XDNNInputDescriptor;
class XDNNOutputDescriptor;
class XDNNEltwiseDescriptor;

/******************************************************
 * Kernel config reader/writer, annotates XBLASHandle
 ******************************************************/

class XDNNKernelConfigInfoMgr {
  public:
#if !XDNNV3
    enum Fields{ N_CORES, DSP_WIDTH, IMG_MEM, VERSION_MAJOR,
      VERSION_MINOR, MODE8BIT, MAX_IMG_WH, MAX_IMG_DEPTH,
      MAX_FILTER_DEPTH, MAX_INSTR_NUM };
#else
    enum Fields{ N_CORES, DSP_WIDTH, AXI_DATA_WIDTH, IMG_MEM, VERSION_MAJOR,
      VERSION_MINOR, MODE8BIT, MAX_INSTR_NUM, MAX_XBAR_ENTRIES };
#endif

    XDNNKernelConfigInfoMgr(std::vector<int> &regVec) 
      : configRegInfo(regVec) {
        if (configRegInfo.size() < 20)
          configRegInfo.resize(20);
      }
    void set(XDNNKernelConfigInfoMgr::Fields fieldIdx, int value) {
      configRegInfo[fieldIdx] = value;
    }
    int get(XDNNKernelConfigInfoMgr::Fields fieldIdx) {
      return configRegInfo[fieldIdx];
    }

  private:
    std::vector<int> &configRegInfo;
};

class XDNNDescPrinter : public XDNNDispatcher{
  public:
    virtual ~XDNNDescPrinter(){}
    virtual void dispatch ( XDNNDataDescriptor * d );
    virtual void dispatch ( XDNNDescriptor * d );
    virtual void dispatch ( XDNNInputDescriptor * d );
    virtual void dispatch ( XDNNOutputDescriptor * d );
};

class XDNNInputDescriptor : public XDNNDataDescriptor {
  public:
    XDNNInputDescriptor() = delete;
    XDNNInputDescriptor( const std::string & layerName,
        int dataTypeSize,
        XDNNTensorShapeType shapeType,
        int n, int c, int h, int w,  size_t hwOffset, size_t hwSizeInBytes,
        bool singleStep) 
      : XDNNDataDescriptor(layerName, dataTypeSize, shapeType, n, c, h, w, hwOffset, hwSizeInBytes , singleStep)
    {
    }
#if XDNNV3
    XDNNInputDescriptor( const std::string & layerName,
        int dataTypeSize,
        XDNNTensorShapeType shapeType,
        int n, int c, int h, int w,  size_t hwOffset, size_t hwSizeInBytes,
        int tiling, int fullSectNum, int replSectNum,
        int replUnitNum, int replUnitWidth, int srcAddrReadFromImgQ,
        bool singleStep) 
      : XDNNDataDescriptor(layerName, dataTypeSize, shapeType, n, c, h, w, hwOffset, hwSizeInBytes, singleStep)
    {
      _tilingOn = tiling;
      _fullSectNum = fullSectNum;
      _replSectNum = replSectNum;
      _replUnitNum = replUnitNum;
      _replUnitWidth = replUnitWidth;
      _srcAddrReadFromImgQ = srcAddrReadFromImgQ;
    }
#endif
    virtual int execute(XComputeUnit * cu);
    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
#if XDNNV3
    bool _tilingOn = false;
    int _fullSectNum=0;
    int _replSectNum=0;
    int _replUnitNum=0;
    int _replUnitWidth=0;
    int _srcAddrReadFromImgQ=0;
#endif

};

class XDNNThroughputInterbufDescriptor : public XDNNDataDescriptor {
  public:
    XDNNThroughputInterbufDescriptor() = delete;
    XDNNThroughputInterbufDescriptor(size_t hwSizeInBytes)
      : XDNNDataDescriptor("", sizeof(short), XDNN_TENSOR_NCHW,
          0, 0, 0, 0, 0, hwSizeInBytes, true)
    {
    }
    virtual ~XDNNThroughputInterbufDescriptor() {}
    virtual int execute(XComputeUnit * cu) {}
    virtual void dispatch ( XDNNDispatcher * v) {}
};

class XDNNOutputDescriptor : public XDNNDataDescriptor {
  public:
    XDNNOutputDescriptor() = delete;
    virtual ~XDNNOutputDescriptor() {}
    XDNNOutputDescriptor( const std::string & layerName,
        int dataTypeSize,
        XDNNTensorShapeType shapeType,
        int n, int c, int h, int w,  size_t hwOffset, size_t hwSizeInBytes,
        bool singleStep) 
      : XDNNDataDescriptor(layerName, dataTypeSize, shapeType, n, c, h, w, hwOffset, hwSizeInBytes , singleStep)
    {
    }

    virtual int execute(XComputeUnit * cu);
    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
};

enum XDNNOperation { XDNN_GATHER, XDNN_SCATTER, XDNN_NOOP, XDNN_CONV, XDNN_CONVDEPTH, XDNN_MAXPOOL, XDNN_AVGPOOL, XDNN_ELTWISE, XDNN_SCALE, XDNN_CONCAT, XDNN_BATCHNORM, XDNN_ACTIVATION, XDNN_MAXPOOLPIPELINED, XDNN_ELTWISE_SCALE, XDNN_THROUGHPUT_INTERBUF, XDNN_UPSAMPLE, XDNN_UNSUPPORTED};

class XDNNCompilerOp 
{
  public:
    XDNNCompilerOp(): _op(XDNN_UNSUPPORTED){ }
    virtual ~XDNNCompilerOp(){}
    XDNNCompilerOp(const std::map<std::string, std::string> &kv);
    XDNNCompilerOp ( const XDNNCompilerOp & op ){
    	_op = op._op;
    	_params = op._params;
    }

    XDNNOperation getOp() const { return _op; }
    const std::map<std::string, std::string> &getParams() const {return _params;}
    int getInt(std::string key) const;
    double getDouble(std::string key) const;
    int getInt(std::string key, int defaultVal) const;
    std::string getStr(std::string key) const;
    std::string getStr(std::string key, const std::string &defaultVal) const;
    bool hasKey(std::string key) const;

    virtual std::string getLayerName() const
	{
    	return _params.find("name")->second;
    }
    virtual bool isInput() const{
    	return false;
    }

    virtual bool isThroughputInterbuf() const{
    	return false;
    }

    virtual bool isOutput() const {
    	return false;
    }
  protected:
    XDNNOperation _op;
    std::map<std::string, std::string> _params;
};

class XDNNInputCompilerOp : public XDNNCompilerOp{
public:
	XDNNInputCompilerOp( const std::map<std::string, std::string> &kv, const std::string & tensorName, const std::string & layerName) : XDNNCompilerOp( kv){
		_tensorName = tensorName;
		_layerName = layerName;
	}
	virtual ~XDNNInputCompilerOp(){}
	virtual bool isInput() const override{
		return true;
	}
	virtual std::string getTensorName() const{
		return _tensorName;
	}

	virtual std::string getLayerName() const override{
		return _layerName;
	}
protected:
	std::string _tensorName, _layerName;
};

class XDNNThroughputInterbufCompilerOp : public XDNNCompilerOp{
public:
	XDNNThroughputInterbufCompilerOp( const std::map<std::string, std::string> &kv ) : XDNNCompilerOp( kv){}
	virtual ~XDNNThroughputInterbufCompilerOp(){}
	virtual bool isThroughputInterbuf() const override{
		return true;
	}
};

class XDNNOutputCompilerOp : public XDNNCompilerOp{
public:
	XDNNOutputCompilerOp( const std::map<std::string, std::string> &kv, const std::string & tensorName, const std::string & layerName) : XDNNCompilerOp( kv){
		_tensorName = tensorName;
		_layerName = layerName;
	}
	virtual ~XDNNOutputCompilerOp(){}
	virtual bool isOutput() const override{
		return true;
	}
	virtual std::string getTensorName() const{
		return _tensorName;
	}
	virtual std::string getLayerName() const override{
		return _layerName;
	}
protected:
	std::string _tensorName, _layerName;
};

class XDNNDescriptor {
  public:
    // default (user manually fills in)
    virtual ~XDNNDescriptor() { }
    XDNNDescriptor(const XDNNCompilerOp &op)
    : _name(op.getParams().at("name"))
    { 
      _cOp = std::shared_ptr <XDNNCompilerOp> ( new XDNNCompilerOp (op)); 
    }

    virtual int execute(XComputeUnit *cu) { return(EXIT_SUCCESS); }

    virtual XDNNOperation OpType() = 0;

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }

    // common args
    std::string _name;
    std::shared_ptr <XDNNCompilerOp> _cOp;

private:
    XDNNDescriptor() : _cOp(NULL) {};
};

class XDNNEltwiseDescriptor : public XDNNDescriptor {
  public:
    XDNNEltwiseDescriptor(const void *A, const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit *cu);
    inline virtual XDNNOperation OpType() { return XDNN_ELTWISE; }
    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }

    const void *_A;
    int _inBaseAddrB;
    bool _perChannelScale;
    bool _add;
    bool _bn;
    bool _relu;
    int _AOffset;
    int _biasOffset;
    int _scaleOffset;
    int _postShiftOffset;
    int _preShiftOffset;
    std::vector<int> _scale;
    std::vector<int> _posScale;
    std::vector<int> _negScale;
    std::vector<int> _postShift;
    std::vector<int> _preShift;
    bool _useQuantParams;
};

class XDNNScatterDescriptor : public XDNNDescriptor {
  public:
    XDNNScatterDescriptor(const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);
    inline virtual XDNNOperation OpType() { return XDNN_SCATTER; }
    //virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
};
class XDNNGatherDescriptor : public XDNNDescriptor {
  public:
    XDNNGatherDescriptor(const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);
    inline virtual XDNNOperation OpType() { return XDNN_GATHER; }
    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
};

class XDNNConvolutionDescriptor : public XDNNDescriptor {
  public: 
    XDNNConvolutionDescriptor(const void *A, int Aoffset, int biasOffset,
      const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);

    inline virtual XDNNOperation OpType() { return XDNN_CONV; }

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }

    const void *_A;
    int _Aoffset;

    // quantization settings
    int _biasOffset;
    int _preShift;
    int _postShift;
    int _scaleVal;
    bool _perChannelScale;

    // crossbar settings
    int _startPtr0;
    int _startPtr1;
    int _endPtr0;
    int _endPtr1;
    int _itnCount0;
    int _itnCount1;
};

#if XDNNV3
class XDNNMaxPoolPipelinedDescriptor : public XDNNConvolutionDescriptor {
  public: 
   XDNNMaxPoolPipelinedDescriptor(const void *A, int Aoffset, int biasOffset,
      const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);

    inline virtual XDNNOperation OpType() { return XDNN_MAXPOOLPIPELINED; }

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
};

#endif

class XDNNConvolutionDepthDescriptor : public XDNNConvolutionDescriptor {
  public: 
    XDNNConvolutionDepthDescriptor(const void *A, int Aoffset, int biasOffset, const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);
    inline virtual XDNNOperation OpType() { return XDNN_CONVDEPTH; }

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }
};

class XDNNPoolDescriptor : public XDNNDescriptor {
  public:
    XDNNPoolDescriptor(const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }

    const void *_A;
    int _AOffset;
    std::vector<int> _scale;
    std::vector<int> _posScale;
    std::vector<int> _negScale;
    std::vector<int> _postShift;
    std::vector<int> _preShift;
    bool _useQuantParams;
};

class XDNNMaxpoolDescriptor : public XDNNPoolDescriptor {
  public: 
    XDNNMaxpoolDescriptor(const XDNNCompilerOp &op)
      : XDNNPoolDescriptor(op) {}
    inline virtual XDNNOperation OpType() { return XDNN_MAXPOOL; }
};
#if XDNNV3
class XDNNUpsampleDescriptor : public XDNNDescriptor {
  public:
    XDNNUpsampleDescriptor(const XDNNCompilerOp &op);

    virtual int execute(XComputeUnit * cu);

    inline virtual XDNNOperation OpType() { return XDNN_UPSAMPLE; }

    virtual void dispatch ( XDNNDispatcher * v) { v->dispatch ( this ); }

};
#endif
class XDNNAvgpoolDescriptor : public XDNNPoolDescriptor {
  public:
    XDNNAvgpoolDescriptor(const XDNNCompilerOp &op)
      : XDNNPoolDescriptor(op) {}
    inline virtual XDNNOperation OpType() { return XDNN_AVGPOOL; }
};

class XDNNDescriptorDB
{
  public:
    XDNNDescriptorDB()= delete;

    virtual ~XDNNDescriptorDB(){}

	XDNNDescriptorDB( const std::string &compileCfg, const std::string & quantizeFile = "", const short * A = nullptr) {
		buildParams(compileCfg);
		if (A) buildDescriptors(A, quantizeFile);
		for (const auto &kv : _compOp) {
			const XDNNCompilerOp * op = kv.second.get();
			std::string layer = kv.first;
			if ( op->isOutput() ) {
				XDNNOutputCompilerOp * out_op = (XDNNOutputCompilerOp*)op;
				std::string cpuTensorName = out_op->getTensorName();
				_outputDesc[cpuTensorName] = std::shared_ptr<XDNNOutputDescriptor>(
						new XDNNOutputDescriptor(out_op->getLayerName(), sizeof(short),
								XDNN_TENSOR_NCHW, 1, op->getInt("inchan"),
								op->getInt("insize_h"), op->getInt("insize_w"),
								op->getInt("ddr_dest"),
								(long long) op->getDouble("ops"), true));
			} else if ( op->isThroughputInterbuf() ){
        _throughputInterbufDesc = std::shared_ptr<XDNNThroughputInterbufDescriptor>(
           new XDNNThroughputInterbufDescriptor((long long)op->getDouble("ops")));
			} else if ( op->isInput() ){
				XDNNInputCompilerOp * in_op = (XDNNInputCompilerOp*)op;
				std::string cpuTensorName = in_op->getTensorName();
				std::string cpuLayerName = in_op->getLayerName();
#if !XDNNV3
				_inputDesc[cpuTensorName] = std::shared_ptr<XDNNInputDescriptor>(
						new XDNNInputDescriptor( cpuLayerName,
								sizeof(short), XDNN_TENSOR_NCHW, 1, op->getInt(
										"inchan"), op->getInt("insize_h"), op->getInt(
										"insize_w"), op->getInt("ddr_src"), (long long)op->getDouble("ops"),true));
#else
				int srcFullSectNum =
						op->hasKey("src_full_sect_num") ?
								op->getInt("src_full_sect_num") :
								ceil(int(op->getInt("inchan")) / 96);
				int srcReplSectNum = op->getInt("src_repl_sect_num",0);
				int srcReplUnitNum = op->getInt("src_repl_unit_num",0);
				int srcReplUnitWidth = op->getInt("src_repl_unit_width",0);
				int srcAddrReadFromImgQ = op->getInt("srcAddrReadFromImgQ");
				int HAT = op->getInt("HAT",0);

				_inputDesc[cpuTensorName] = std::shared_ptr<XDNNInputDescriptor>(
						new XDNNInputDescriptor( cpuLayerName, sizeof(short), XDNN_TENSOR_NCHW,
								1, op->getInt("inchan"), op->getInt("insize_h"),
								op->getInt("insize_w"), op->getInt("ddr_src"),
								(long long) op->getDouble("ops"), HAT,
								srcFullSectNum, srcReplSectNum, srcReplUnitNum,
								srcReplUnitWidth, srcAddrReadFromImgQ,
								/*singleStep=*/true));
#endif
			}
		}
	}

    XDNNDescriptorDB(XDNNDescriptorDB const& inst) = delete;

    const std::vector<std::string>& getLayerNames() const{
      return _activeLayers;
    }

    std::shared_ptr<XDNNDescriptor> getDescriptor ( const std::string & layerName) const;
    XDNNOperation getLayerType ( const std::string & layerName ) const;
    XDNNCompilerOp getLayerJsonParams(const std::string &layerName) const;

    void execute ( XComputeUnit * cu )
    {
        for (int si = 0; si < _activeLayers.size(); si++)
            _desc[_activeLayers[si]]->execute(cu);
    }

    size_t getInputSize() const{
    	size_t sz = 0;
    	for (const auto& kv : _inputDesc) {
    		sz += kv.second->getSize();
    	}
    	return sz;
    }
    size_t getOutputSize() const{
    	size_t sz = 0;
    	for (const auto& kv : _outputDesc) {
    		sz += kv.second->getSize();
    	}
    	return sz;
    }

    size_t num_desc() const{
    	return _desc.size();
    }

    const std::map< std::string, std::shared_ptr <XDNNInputDescriptor> >& getInputDesc() const
    {
    	return _inputDesc;
    }

    const XDNNThroughputInterbufDescriptor *getThroughputInterbufDesc() const {
      return _throughputInterbufDesc.get();
    }

    const std::map< std::string, std::shared_ptr <XDNNOutputDescriptor> >& getOutputDesc() const
    {
    	return _outputDesc;
    }

    const std::unordered_map < std::string, std::shared_ptr<XDNNDescriptor> > & getDesc() const
    {
    	return _desc;
    }

  private:
    void buildDescriptors ( const short * A, const std::string &quantizeFile);
    std::vector<std::string> split(const std::string &s, char delim);
    void buildParams (const std::string & fileName);
    void buildParamsFromJson (const std::string & fileName);

    std::vector<std::string> _activeLayers;
    std::unordered_map< std::string, std::shared_ptr<XDNNCompilerOp> > _compOp;
    std::map< std::string, std::shared_ptr <XDNNInputDescriptor> > _inputDesc;
    std::shared_ptr<XDNNThroughputInterbufDescriptor> _throughputInterbufDesc;
    std::map< std::string, std::shared_ptr <XDNNOutputDescriptor> > _outputDesc;
    std::unordered_map < std::string, std::shared_ptr<XDNNDescriptor> > _desc;
};

class XBLASHandle;
class XBLASConfig;
class XMemPtr;

class XDNNQuantBuf{
public:
	XDNNQuantBuf( unsigned bsz, unsigned inSize, unsigned outSize)
	{
		_qInBuf.resize(bsz);
		_qOutBuf.resize(bsz);
		for (unsigned i = 0; i < bsz; i++){
			_qInBuf[i] = new short[inSize];
			_qOutBuf[i] = new short[outSize];
		}
		_insz = inSize;
		_outsz = outSize;
	}
	~XDNNQuantBuf(){
		for (unsigned i = 0; i < _qInBuf.size(); i++){
			delete [] _qInBuf[i];
			delete [] _qOutBuf[i];
		}
	}

	short* getInBuf(unsigned idx) const
	{
		assert ( idx < _qInBuf.size());
		return _qInBuf[idx];
	}

	short* getOutBuf(unsigned idx) const{
		assert ( idx < _qOutBuf.size());
		return _qOutBuf[idx];
	}

	unsigned getInSize() const { return _insz;}
	unsigned getOutSize() const { return _outsz;}
protected:

	std::vector<short*> _qInBuf, _qOutBuf;
	unsigned _insz, _outsz;
};

/*
 * @brief Models a scripted executor. Defines an abstraction that includes: Netfile(script) + CU-set + weights +
 * Buffers. CU-set is any subset of the number of CUs in the FPGA. Execution can be blocking or non-blocking. In
 * non-blocking mode, a future is obtained and saved in the state. Clients have to waitForResults(). Each CU-set will
 * get its own stream-Id. CU-set has to be disjoint across all XDNNScriptExecutor instances. No checks for this yet.
 *
 * CU-set is a bit-mask, with Bit-i indicating if CU-i is to be used. CU-set = 0 implies all CUs can be used
 *
 */
template <typename Dtype>
class XDNNScriptExecutor
{
  public:
    XDNNScriptExecutor () = delete;
    XDNNScriptExecutor ( const std::vector<XBLASHandle*> & xblas_handles,
                         const std::map<std::string, XDNNWeightParam<Dtype> > &m,
                         const std::string & xdnn_net_file,
                         const std::string & quantize_CfgFile,
                         float scaleB, unsigned int cu_mask = 0 );

    XDNNScriptExecutor ( const std::vector<XBLASHandle*> & xblas_handles,
                         const std::string & weights_dir,
                         const std::string & xdnn_net_file,
                         const std::string & quantize_CfgFile,
                         float scaleB, unsigned int cu_mask = 0 );

    // This is a simplified API for python. Expects loadBlobToDdr to have happened already
    XDNNScriptExecutor ( const std::vector<XBLASHandle*> & xblas_handles,
                         short * wgt_ptr,
                         const std::string & xdnn_net_file,
                         const std::string & quantize_CfgFile,
                         float scaleB,
                         unsigned int cu_mask = 0 );

    virtual ~XDNNScriptExecutor( ){ }

    short* XDNNLoadWeights(const std::vector<XBLASHandle*> & xblas_handles,
        const std::string & weights_dir,
        const std::string & xdnn_net_file, const std::string & quantize_CfgFile, unsigned int cuMask);
        
    short* XDNNLoadWeights(const std::vector<XBLASHandle*> & xblas_handles, const std::map<std::string, XDNNWeightParam<Dtype> > & m,
        const std::string & xdnn_net_file, const std::string & quantize_CfgFile, unsigned int cuMask);

    const std::vector<std::string> & getLayerNames() const { return _descDB->getLayerNames(); }
    std::shared_ptr<XDNNDescriptor> getDescriptor ( const std::string & layerName ) const { return _descDB->getDescriptor(layerName); };

    //virtual void execute ( const std::vector <const Dtype*> & input_ptrs, Dtype *&output, int streamId, bool blocking);
    virtual void execute ( const  std::unordered_map <std::string, std::vector<const Dtype*> > & input_ptrs,
    					   std::unordered_map< std::string, std::vector<Dtype*>> & output, int streamId);
    virtual void execute_async ( const  std::unordered_map <std::string, std::vector<const Dtype*> > input_ptrs,
    					         std::unordered_map< std::string, std::vector< Dtype*>> output, int streamId);

    virtual void waitForResults (int streamId=0);

    virtual float readHardwareCounter(int devIdx, int cuIdx);

  protected:
    void loadInstructions(XBLASHandle * handle, XComputeUnit *cu);
    void init (  const std::vector<XBLASHandle*> & xblas_handles,
                 const std::string & xdnn_net_file,
                 const std::string & quantize_CfgFile,
                 short * wgt_ptr,
                 float scaleB,
                 unsigned int cuMask );

    int fpgaXdnn(XBLASHandle &handle,
		const std::unordered_map <std::string, std::vector<const Dtype*> > & in_blob,
		std::shared_ptr<XDNNQuantBuf> & qbuf, unsigned batch_id,
        int streamId);

    int getNextPrefCuIdx ();
    void XDNNWaitAndPostProcess (int streamId,
    							 unsigned bsz,
								 std::unordered_map< std::string, std::vector<Dtype*>> output);

    std::vector<XBLASHandle*> _xblas_handles;
    std::string _compiler_file;
    std::string _quantize_cfg_file;
    std::unordered_map< int, std::unordered_map<int, std::shared_ptr<XDNNQuantBuf> > > _quantizedBuffers;
    float _scaleB;
    short * _wgt_ptr;

    std::shared_ptr <XDNNDescriptorDB> _descDB;

    unsigned int _cuMask;   // Bitmask for allowed CUs. 0=>all.
    std::vector <int> _cuIdxVec;
    unsigned int _cuIdxCntr;
    std::map<int, std::future<void> > _streamFutures;
    XDNNLayerQuantParam *_startQp, *_lastQp;
    std::unordered_map<std::string, XDNNLayerQuantParam*> _quantparam;
    unsigned _bitWidth;
    int _startIdx, _stopIdx, _dflStartIdx, _dflStopIdx;
};

int XDNNInitScript(XComputeUnit *cu);
void XDNNScriptExecute(XPipelinePacket *packet);
std::vector<int> XDNNUpdateStatusAndGetCompleted(
        XBLASHandle* h, int kIdx, int cuIdx);

void XDNNMatToDDR(short *src,
    unsigned short kern_w, unsigned short  kern_h, unsigned int inChans, unsigned int outChans,
    short *&out, unsigned int &outRows, unsigned int &outCols);

void XDNNV3LoadCfgData(uint64_t* conv_och_cfg, unsigned int outChans, int *&out);

template <class SrcT>
void XDNNV3MatToDDR(SrcT *matrix_filter,uint64_t* conv_och_cfg,
    unsigned short kern_w, unsigned short  kern_h, unsigned int inChans, unsigned int outChans,
    int *&out, int bitWidth, int srcFullSectNum, int srcReplSectNum, int srcReplUnitNum, int srcReplUnitWidth, bool convHalfRateMode, int slice, std::vector<int> &breakIdx );

void XDNNGetMatToDDRSize(unsigned short kern_w, unsigned short kern_h, unsigned int inChans, unsigned int outChans,
    unsigned int &outRows, unsigned int &outCols);

template<typename T>
void XDNNdbgDumpToFile(const T *data, int size, std::string fname);
template<typename T>
void XDNNdbgReadFromFile(T *data, int size, std::string fname);

std::vector<XDNNLayerQuantParam> XDNNloadLayerQuantizationParams(std::string fname);
short XDNNQuantize(float in, float thresh, int bitWidth, bool doRounding=false);
float XDNNUnQuantize(short in, float thresh, int bitWidth, int halfRange=0);
void XDNNUnQuantize(short *in, float *out, int sz, float thresh, int bitWidth);
void XDNNQuantizePack(short *in, int sz, int n);
void XDNNQuantizeUnpack(char *in, int sz, int n);
void XDNNQuantizeCompress(short *in, int sz);

template<typename DType>
int XDNNV3FillWeightsBiasQuantBlob(short *blob, int offset,
    std::string layerName, std::string fpgaCfgFile,
    const DType *weight, unsigned int weight_sz, float scale_wgt,
    const DType *bias, unsigned int bias_sz, float scale_bias,
    unsigned short kern_w, unsigned short kern_h,
    unsigned int inChans, unsigned int outChans, int srcFullSectNum,
    int srcReplSectNum, int srcReplUnitNum, int srcReplUnitWidth,
    bool convHalfRateMode, std::string opType, int slice);
template<typename DType>
int XDNNFillWeightsBiasQuantBlob(short *blob, int offset,
    std::string layerName, std::string fpgaCfgFile,
    const DType *weight, unsigned int weight_sz, float scale_wgt,
    const DType *bias, unsigned int bias_sz, float scale_bias,
    unsigned short kern_w, unsigned short kern_h,
    unsigned int inChans, unsigned int outChans, XDNNOperation op);

void XDNNPostProcessOutput(std::string layerName, void *output,
    int numBatches, int batchImgSize, std::string fpgaCfgFile,
    float scale);
std::vector<int>
XDNNComputeWeightsBiasQuantByteOffsets(int &baseOffset,
    int kernWidth, int kernHeight, int inChans, int outChans,
    bool doQuant, XDNNOperation op_type=XDNN_CONV);
std::vector<int>
XDNNV3ComputeWeightsBiasQuantByteOffsets(int &baseOffset,
    int kernWidth, int kernHeight, int inChans, int outChans,
    bool is8bit, bool convHalfRateMode);
void XDNNMaskToIndex (std::vector <int>& cuIdxVec, unsigned int cuMask);

void XDNNUpdateHardwareCounters(XPipelinePacket* packet);
void XDNNUpdateErrorStatus(XPipelinePacket* packet);

extern "C" {
  int XDNNV3ComputeWeightsBiasQuantSize(int kernWidth, int kernHeight, int outChans, int srcFullSectNum, int srcReplSectNum, int srcReplSectUnitNum, bool is8bit,
  bool convHalfRateMode);
  int XDNNComputeWeightsBiasQuantSize( int kernWidth, int kernHeight, int inChans, int outChans, bool doQuant, int op);
  short* XDNNMakeWeightsBiasQuantBlob(int totalSize);
  int XDNNV3FillWeightsBiasQuantBlob(short *blob, int offset,
      char *layerName, char *fpgaCfgFile,
      const float *weight, unsigned int weight_sz, float scale_wgt,
      const float *bias, unsigned int bias_sz, float scale_bias,
      unsigned short kern_w, unsigned short kern_h,
      unsigned int inChans, unsigned int outChans, int srcFullSectNum,
      int srcReplSectNum, int srcReplUnitNum, int srcReplUnitWidth,
      bool convHalfRateMode, char *opType, int slice);
  int XDNNFillWeightsBiasQuantBlob(short *blob, int offset,
      char *layerName, char *fpgaCfgFile,
      const float *weight, unsigned int weight_sz, float scale_wgt,
      const float *bias, unsigned int bias_sz, float scale_bias,
      unsigned short kern_w, unsigned short kern_h,
      unsigned int inChans, unsigned int outChans, XDNNOperation op);

  void* XDNNMakeScriptExecutor(XBLASHandle **handlePtrs, int numHandles,
      short *weight, char *xdnnNetFile, char *fpgaCfgFile, float scaleB,
      unsigned int cuMask=0);
  void* XDNNMakeScriptExecutorAndLoadWeights(XBLASHandle **handlePtrs, 
      int numHandles, char *weightsPath, char *xdnnNetFile, char *fpgaCfgFile, 
      float scaleB, unsigned int cuMask=0);
  void* XDNNMakeScriptExecutorAndLoadWeightsFromMem(
      XBLASHandle **handlePtrs, int numHandles, const int numWeightLayers,
      const char **weightLayerNames, const float **weights, const int *weightsSz, 
      const float **bias, const int *biasSz,
      char *xdnnNetFile, char *fpgaCfgFile, 
      float scaleB, unsigned int cuMask=0);
  void XDNNExecute_2D_float(void *scriptExecutor_, const float ***fpgaInputAddr, char ** inNames, unsigned numInput,
		  	  	  	  	  	float ** output, unsigned * output_size, char ** outNames, unsigned numOutput,
							unsigned bsz, int streamId, bool blocking=true);

  void XDNNWaitForResults(void *scriptExecutor_, int streamId=0);
  float XDNNReadHardwareCounter(void *scriptExecutor_, int devIdx=0, int cuIdx=0);
  void XDNNReadWeightsFile(char *fname, char **layerName, int **kernWidth, int **kernHeight,
      int **inChans, int **outChans, int **numVals, float **vals);
  char XDNNQuantizeAvgPool(float sumQuantizedVal, float scaleVal, int postShift, int bitWidth);
  void XDNNQuantizeTensor(float threshIn, int bitWidth, float *vals, int numVals);
  void XDNNQuantizeWeights(float thresh, int bitWidth, float *vals, int numVals);
  short XDNNQuantizeBias(float threshOut, int bitWidth, float val);
  void XDNNQuantizeBiasV3(const float *in, int* out, int sz, float threshIn, std::vector<float>* threshparams, int bitWidth, bool doRounding);
  void XDNNQuantizeBiasV3_scale(const float *in, int *out, int sz, float threshOut, std::vector<int>* postShift, int bitWidth, bool doRounding);
 int XDNNV3QuantizeBias(float threshIn, float threshParams, int bitWidth, float valIn, bool doRounding = true);
  void XDNNV3QuantizeInterLayer(int preShift, int scale, int postShift, int bitWidth, long long *vals, int numVals, int bias);
  void XDNNQuantizeInterLayer(int preShift, int scale, int postShift, int bitWidth, long long *vals, int numVals);
  void XDNNUnQuantizeTensor(float threshOut, int bitWidth, float *vals, int numVals);
  void XDNNFetchBatchBlob(short* batchBlob, int sizeToRead, char* fname);

} // extern "C" {

#endif // XDNN_H

