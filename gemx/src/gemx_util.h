/*
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _GEMX_UTIL_H
#define _GEMX_UTIL_H

#include <chrono>
#include <assert.h>
#include <boost/assert.hpp>
#include <thread>
using namespace std;

#define XASSERT(cond, msg) {\
    if(!(cond))\
    {\
        std::stringstream str;\
        str << msg;\
        BOOST_ASSERT_MSG(cond, str.str().c_str());\
    }\
}

namespace gemx{

string getThreadIdStr()
{
    return boost::lexical_cast<std::string>(std::this_thread::get_id());
}

unsigned long getThreadId(){
    std::string threadId = getThreadIdStr();
    //unsigned long threadNumber =
    //sscanf(threadId.c_str(), "%lx", &threadNumber);
    return std::stol(threadId);
}



class XTimer
{
  public:
    XTimer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
      return chrono::duration_cast<second_>
        (clock_::now() - beg_).count(); }

  private:
    typedef chrono::high_resolution_clock clock_;
    typedef chrono::duration<double, ratio<1> > second_;
    chrono::time_point<clock_> beg_;
};


// Matrix descriptor with data itself stored in caller's space
template<typename T>
class Mat {
private:
    unsigned int m_Rows, m_Cols, m_Ld, m_buf_sz;
    bool m_ownmem;
    T *m_Addr;
public:
    const static size_t GEMX_CMP_WIDTH = 11;
    Mat() = delete;
    ~Mat() {
        if (m_ownmem && m_Addr) {
            free(m_Addr);
        }
    }
    Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld) :
        m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_ownmem(true) {
        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
        posix_memalign((void**) &m_Addr, 4096, m_buf_sz);
        //m_Addr = (T*)aligned_alloc ( 4096, sizeof(T) * p_Rows * p_Cols);
    }

    Mat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T *p_Addr) :
        m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr), m_ownmem(
                false) {
        m_buf_sz = sizeof(T) * p_Rows * p_Ld;
    }
    Mat& operator=(const Mat& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < m_Ld; ++col) {
                m_Addr[row][col] = p_Src.getVal(row, col);
            }
        }
        return *this;
    }

    unsigned int buf_sz(){
        return m_buf_sz;
    }
    T*& data() {
        return m_Addr;
    }

    inline T &getVal(unsigned int p_Row, unsigned int p_Col) {
        return m_Addr[p_Row * ld() + p_Col];
    }
    inline unsigned int rows() {
        return m_Rows;
    }
    inline unsigned int cols() {
        return m_Cols;
    }
    inline unsigned int ld() {
        return m_Ld;
    }

    void init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld,
            T *p_Addr) {
        m_Rows = p_Rows;
        m_Cols = p_Cols;
        m_Ld = p_Ld;
        m_Addr = p_Addr;
    }

    void fillModRange(T p_Min, T p_Max) {
        T l_val = p_Min;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val++;
                if ( l_val > p_Max ) l_val = p_Min;
            }
        }
    }

    void fillMod(T p_Max, T p_First = 0) {
        T l_val = p_First;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val;
                l_val++;
                l_val %= p_Max;
            }
        }
    }

    void multiply(Mat & p_A, Mat & p_B) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                int64_t l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                    l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                }
                getVal(row, col) = (T)l_val;
            }
        }
    }

        void
    multiplyAddScale(Mat & p_A, Mat & p_B,  Mat<int> & p_X, int postScaleVal, int postScaleShift) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
                assert(p_X.rows() == rows());
                assert(p_X.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
          for (unsigned int col = 0; col < cols(); ++col) {
            int64_t l_val = 0;
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
              l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
            }
        l_val += p_X.getVal(row, col);
                l_val = (l_val >> postScaleShift ) * postScaleVal;
                getVal(row, col) = (T)(l_val);
          }
        }
      }

    void matMultWithScaleAndPRelu(Mat & p_A, Mat & p_B, Mat<int> & p_X,  int32_t p_postScale, int16_t p_PReluVal) {
        cout << "A rows: " << p_A.rows() << " this rows: " << rows() << endl;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        assert(p_X.rows() == rows());
        assert(p_X.cols() == cols());
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                int64_t l_val = 0;
                for (unsigned int k = 0; k < p_A.cols(); ++k) {
                    l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
                }

                //                      if ((row == 2) && (col == 0)) {
                //                          bitset<64> l_bVal{l_val};
                //                          cout << "C[2,0]= " << l_bVal << "\n";
                //                      }
                l_val += p_X.getVal(row,col);
                unsigned int l_psShift = p_postScale & 0x00ff;
                unsigned int l_psVal = p_postScale >> 8;
                l_val = (l_val >> l_psShift) * l_psVal;
                T l_entry = (T)(l_val);
                if (l_entry < 0) {
                    l_entry = (l_entry  >> (p_PReluVal & 0x003f))* (T)(p_PReluVal >> 6);
                }
                getVal(row, col) = l_entry;
            }
        }
    }

    void multiplyGf(Mat & p_A, Mat & p_B, unsigned int p_EdgeWidth) {
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        cout << "  DEBUG multiplyGf rows=" << rows() << "  cols=" << cols()
                                << "\n";
        for (unsigned int rowBlock = 0; rowBlock < rows() / p_EdgeWidth;
                ++rowBlock) {
            for (unsigned int colBlock = 0; colBlock < cols() / p_EdgeWidth;
                    ++colBlock) {
                for (unsigned int row = 0; row < rows(); ++row) {
                    for (unsigned int col = 0; col < cols(); ++col) {
                        T l_val = 0;
                        for (unsigned int k = 0; k < p_A.cols(); ++k) {
                            l_val += p_A.getVal(k + rowBlock * p_EdgeWidth,
                                    col + colBlock * p_EdgeWidth)
                                                    * p_B.getVal(k + rowBlock * p_EdgeWidth,
                                                            col + colBlock * p_EdgeWidth);
                        }
                        getVal(row + rowBlock * p_EdgeWidth,
                                col + colBlock * p_EdgeWidth) = l_val;
                        cout << "DEBUG multiplyGf after k-loop " << *this
                                << "\n";
                    }
                }
            }
        }
    }
    // Matrix A is in GvA format (also dimensions are wider and shorter)
    // The p_rowEdgeWidth just inficates the compute array intake edge to allow for matrix dimension adjustment
    void multiplyGemvGf(Mat & p_A, Mat & p_B, unsigned int p_rowEdgeWidth) {
        assert(p_A.rows() * p_rowEdgeWidth == rows());
        assert(p_A.cols() == p_B.rows() * p_rowEdgeWidth);
        assert(p_B.cols() == cols());
        cout << "  DEBUG multiplyGvA format rows=" << rows() << "  cols="
                << cols() << "\n";
        // Rows here are mblocks, cols are within the mblock
        for (unsigned int row = 0; row < p_A.rows(); ++row) { // A is already in block format
            for (unsigned int col = 0; col < p_A.cols(); ++col) {
                unsigned int k = col / p_rowEdgeWidth;
                unsigned int w = col % p_rowEdgeWidth;
                T l_a = p_A.getVal(row, col);
                T l_b = p_B.getVal(k, 0);
                getVal(w + row * p_rowEdgeWidth, 0) += l_a * l_b;
                //cout << "        += a * b  = " << l_a << " * " << l_b << "\n";
            }
            //cout << "    DEBUG multiplyGemvGf after k-loop " << *this << "\n";
        }
    }
#if 0
    void
    multiplySpmv(SpMat<T, TspD, Tsp> & p_A, Mat & p_B) {
        T l_val = 0;
        assert(p_A.rows() == rows());
        assert(p_A.cols() == p_B.rows());
        assert(p_B.cols() == cols());
        vector<MtxRow> l_rows = p_A.getNnzVector();
        for (MtxRow &l_row : l_rows) {
            unsigned int row = l_row.getRow(),
                    col = l_row.getCol();
            double l_val = l_row.getVal();
            getVal(row, 0) += l_val * p_B.getVal(col, 0);
            //cout << "DEBUG multiplySpmv row=" << row << " col=" << col << "  "
            //          << l_val << " * " << p_B.getVal(col, 0)
            //          << " was added to " << getVal(row, 0) << "\n";
        }
    }
#endif

    void transpose(Mat & p_A) {
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                getVal(row, col) = p_A.getVal(col, row);
            }
        }
        swap(m_Rows, m_Cols);
    }
    void transposeGva(Mat & p_A, unsigned int p_rowEdgeWidth,
            unsigned int p_colEdgeWidth) {
        unsigned int l_pos = 0;
        for (unsigned int rowBlock = 0; rowBlock < p_A.rows() / p_rowEdgeWidth;
                ++rowBlock) {
            for (unsigned int colBlock = 0;
                    colBlock < p_A.cols() / p_colEdgeWidth; ++colBlock) {
                for (unsigned int col = 0; col < p_colEdgeWidth; ++col) {
                    for (unsigned int row = 0; row < p_rowEdgeWidth; ++row) {
                        getVal(l_pos / cols(), l_pos % cols()) = p_A.getVal(
                                row + rowBlock * p_rowEdgeWidth,
                                col + colBlock * p_colEdgeWidth);
                        l_pos++;
                    }
                    //cout << "DEBUG transposeGva step " << *this << "\n";
                }
            }
        }
        swap(m_Rows, m_Cols);
    }
    void print(ostream& os) {
        os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
        unsigned int l_cols = cols(); // normal matrix
        //ld();; // parent matrix (within Ld
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < l_cols; ++col) {
                os << int(getVal(row, col)) << " ";
            }
            os << "\n";
        }
    }
    bool cmp(float p_TolRel, float p_TolAbs, Mat &p_Ref) {
        bool ok = true;
        unsigned int l_verbose = 1; // 0 none, 1 if not exactly equal, 2 if passed tolerance, 3 show all
        unsigned int l_numExactMatches = 0, l_numMismatches = 0;
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                string l_Prefix = "      row " + to_string(row) + " col "
                        + to_string(col);
                T v = getVal(row, col);
                T vRef = p_Ref.getVal(row, col);
                bool l_exactMatch = false;
                bool l_ok = cmpVal(p_TolRel, p_TolAbs, vRef, v, l_Prefix,
                        l_exactMatch, 1);
                ok = ok && l_ok;
                if (l_exactMatch) {
                    l_numExactMatches++;
                }
                if (!l_ok) {
                    l_numMismatches++;
                }
            }
        }
        unsigned int l_total = rows() * cols();
        unsigned int l_withinTolerance = l_total - l_numExactMatches
                - l_numMismatches;
        cout << "  Compared " << l_total << " values:" << "  exact match "
                << l_numExactMatches << "  within tolerance "
                << l_withinTolerance << "  mismatch " << l_numMismatches
                << "\n";
        return (ok);
    }

    bool cmpVal(float p_TolRel, float p_TolAbs, T vRef, T v,
            string p_Prefix, bool &p_exactMatch, unsigned int p_Verbose) {
        float l_diffAbs = abs(v - vRef);
        float l_diffRel = l_diffAbs;
        if (vRef != 0) {
            l_diffRel /= abs(vRef);
        }
        p_exactMatch = (vRef == v);
        bool l_status = p_exactMatch || (l_diffRel <= p_TolRel)
                                || (l_diffAbs <= p_TolAbs);
        if ((p_Verbose >= 3) || ((p_Verbose >= 2) && !p_exactMatch)
                || ((p_Verbose >= 1) && !l_status)) {
            cout << p_Prefix << "  ValRef " << left
                    << setw(GEMX_CMP_WIDTH) << vRef << " Val " << left
                    << setw(GEMX_CMP_WIDTH) << v << "  DifRel "
                    << left << setw(GEMX_CMP_WIDTH) << l_diffRel
                    << " DifAbs " << left << setw(GEMX_CMP_WIDTH)
            << l_diffAbs << "  Status " << l_status << "\n";
        }
        return (l_status);
    }

};
}



#endif
