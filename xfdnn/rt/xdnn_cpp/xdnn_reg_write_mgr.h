// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//
/*
 * xdnn_reg_write_mgr.h
 *
 *  Created on: Apr 3, 2019
 *      Author: vincentm
 */

#ifndef XDNN_REG_WRITE_MGR_H_
#define XDNN_REG_WRITE_MGR_H_
#define XDNN_ONEHACK 0

#if XDNN_ONEHACK
#include <vector>
#include "CL/cl_ext_xilinx.h"
#include "experimental/xrt++.hpp"
#endif

struct xrt_device;

namespace xdnn {

class RegisterWriteManager {

#if XDNN_ONEHACK
	using addr_type = xrtcpp::addr_type;
	using value_type = xrtcpp::value_type;
	using pairs_type = std::pair<addr_type, value_type>;
	using pairs_container = std::vector<pairs_type>;
	using exec_write_cmd = xrtcpp::exec::exec_write_command;

private:

	pairs_container _mVectorAddrValuePairs; // compact space for performance
	bool _mbClearOnWait; // clear vector on Execute
	unsigned int _miNumValidElements; // number of valid elements
	static unsigned int _miMaxValidElements; // max of valid elements
	// TODO: look at shared_ptr for early destructions
	std::shared_ptr<exec_write_cmd> _mcmd; // cmd
	int _miCUidx; // CU idx

	// temp, assume memory allocated elsewhere
	xrt_device* _mxdev;

	// _mbClearOnWait
	bool getClearOnWait() const;
	// _miNumValidElements
	unsigned int getNumValidElements() const;
	void setNumValidElements(unsigned int value);
	void incNumValidElements(unsigned int value = 1);
	// _miMaxValidElements
	unsigned int getMaxValidElements() const;
	void setMaxValidElements(unsigned int value);
	// _mcmd
	exec_write_cmd* getCmd() const;

	// xrt_device
	xrt_device* getXdev() const;
	void setXdev(xrt_device* value);

	// CU idx
	int getCUidx() const;
	void setCUidx(int value);

	// TODO: Put into utils/tools
	static unsigned int BitCount(unsigned int value);
#endif

	//C++14
#if 0
	static constexpr unsigned int myBitCount() {
		// max number of pairs per packet
		ert_packet pkt;
		pkt.count = 0;
		pkt.count = ~pkt.count;
		unsigned int size = BitCount(pkt.count);
		return size;
	}
#endif

#if XDNN_ONEHACK
	const pairs_container& getPairs() const;
	pairs_container& getPairs();
	void clearPairs();
#endif

public:

	RegisterWriteManager() = delete;
	explicit RegisterWriteManager(xrt_device* xdev);
	virtual ~RegisterWriteManager();

#if XDNN_ONEHACK
	// TODO: provide status to USER? (e.g. return bool)
	void setClearOnWait();
	void resetClearOnWait();
	void toggleClearOnWait();
	void setClearOnWait(bool value);

	// functions
	bool addAddrValuePair(const addr_type addr, const value_type value);
	void execute(value_type cuidx);
	void wait();
#endif

};

}// namespace
#endif /* XDNN_REG_WRITE_MGR_H_ */
