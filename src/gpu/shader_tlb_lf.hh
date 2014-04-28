/*
 * Copyright (c) 2011 Mark D. Hill and David A. Wood
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SHADER_TLB_HH_
#define SHADER_TLB_HH_

#include <map>
#include <set>

#include "arch/x86/tlb.hh"
#include "base/statistics.hh"
#include "params/ShaderTLB.hh"
#include "sim/tlb.hh"

class ShaderMMU;
class CudaGPU;

class GPUTlbEntry {
public:
    Addr vpn;
    Addr ppn;
    bool free;
    Tick mruTick;
    uint32_t hits;
    GPUTlbEntry() : vpn(0), ppn(0), free(true), mruTick(0), hits(0) {}
    void setMRU() { mruTick = curTick(); }
};

//Class added for Prefetch Buffer entry
class GPUPBEntry {
public:
    Addr vpn;
    Addr ppn;
    bool free; //Valid
    Tick mruTick;
    uint32_t L1PBhits;
    uint32_t TLB_ID;
    GPUPBEntry() : vpn(0), ppn(0), free(true), mruTick(0), L1PBhits(0) {}
    void setMRU() { mruTick = curTick(); }
};

class BaseTLBMemory {
public:
    virtual bool lookup(Addr vpn, Addr& ppn, bool set_mru=true) = 0;
    virtual void insertL1TLB(ShaderTLB *tlb, Addr vpn, Addr ppn, bool Prefetch=false) = 0;
};

class TLBMemory : public BaseTLBMemory {
    int numEntries;
    int sets;
    int ways;

    GPUTlbEntry **entries;

protected:
    TLBMemory() {}

public:
    TLBMemory(int _numEntries, int associativity) :
        numEntries(_numEntries), sets(associativity)
    {
        if (sets == 0) {
            sets = numEntries;
        }
        assert(numEntries % sets == 0);
        ways = numEntries/sets;
        entries = new GPUTlbEntry*[ways];
        for (int i=0; i < ways; i++) {
            entries[i] = new GPUTlbEntry[sets];
        }
    }
    ~TLBMemory()
    {
        for (int i=0; i < sets; i++) {
            delete[] entries[i];
        }
        delete[] entries;
    }

    virtual bool lookup(Addr vpn, Addr& ppn, bool set_mru=true);
    virtual void insertL1TLB(ShaderTLB *tlb, Addr vpn, Addr ppn, bool Prefetch=false);
    virtual void insert(Addr vpn, Addr ppn);
};

class InfiniteTLBMemory : public BaseTLBMemory {
    std::map<Addr, Addr> entries;
public:
    InfiniteTLBMemory() {}
    ~InfiniteTLBMemory() {}

    bool lookup(Addr vpn, Addr& ppn, bool set_mru=true)
    {
        auto it = entries.find(vpn);
        if (it != entries.end()) {
            ppn = it->second;
            return true;
        } else {
            ppn = Addr(0);
            return false;
        }
    }
    void insertL1TLB(ShaderTLB *tlb, Addr vpn, Addr ppn, bool Prefetch=false)
    {
        entries[vpn] = ppn;
    }

    void insert(Addr vpn, Addr ppn)
    {
    	entries[vpn] = ppn;
    }
};

class ShaderTLB : public BaseTLB
{
private:
    unsigned numEntries;

    Cycles hitLatency;

    // Pointer to the SPA to access the page table
    CudaGPU* cudaGPU; //-should retain this here only
    bool accessHostPageTable;

    BaseTLBMemory *tlbMemory;

    void translateTiming(RequestPtr req, ThreadContext *tc,
                         Translation *translation, Mode mode);

    ShaderMMU *mmu;
    //CudaGPU* cudaGPU;


public:

    typedef ShaderTLBParams Params;
    ShaderTLB(const Params *p);

    // For checkpoint restore (empty unserialize)
    virtual void unserialize(Checkpoint *cp, const std::string &section);

    void beginTranslateTiming(RequestPtr req, BaseTLB::Translation *translation,
                              BaseTLB::Mode mode);

    void finishTranslation(Fault fault, RequestPtr req, ThreadContext *tc,
                           Mode mode, Translation* origTranslation);

    void demapPage(Addr addr, uint64_t asn);
    void flushAll();

    void insert(Addr vpn, Addr ppn);

    void regStats();

    Stats::Scalar numL1Prefetches; //Sharmila
    Stats::Scalar L1prefetchHits; //Sharmila
    Stats::Scalar L1TLBhits;
    Stats::Scalar L1TLBmisses;
    Stats::Formula hitRate;
    Stats::Formula PBhitRate;

	// Log the vpn of the access. If we detect a pattern issue the prefetch
    // This is currently just a simple 1-ahead prefetcher
    void tryL1Prefetch(Addr vpn, ThreadContext *tc);

    // Insert prefetch into prefetch buffer
    void insertL1Prefetch(Addr vpn, Addr ppn, uint32_t TLB_ID);
   
    //New class added by Sharmila
    class L2TranslationRequest //: public BaseTLB::Translation
    {
    public:
        ShaderTLB *TLB;
        BaseTLB::Translation *wrappedTranslation; //Should change this
        RequestPtr req;
        BaseTLB::Mode mode;
        ThreadContext *tc;
        Addr vpn;
        Cycles beginFault;
        //Cycles beginWalk;
        bool prefetch;

    public:
        L2TranslationRequest(ShaderTLB *_tlb,
                           BaseTLB::Translation *translation, RequestPtr _req,
                           BaseTLB::Mode _mode, ThreadContext *_tc,
                           bool prefetch=false);
        void markDelayed() { wrappedTranslation->markDelayed(); }
        /*
        void finish(Fault fault, RequestPtr _req, ThreadContext *_tc,
                    BaseTLB::Mode _mode)
        {
            assert(_mode == mode);
            assert(_req == req);
            assert(_tc == tc);
            mmu->finishWalk(this, fault); //Change this to another function
        }
        void walk(X86ISA::TLB *walker) {
            beginWalk = mmu->curCycle();
            assert(walker != NULL);
            pageWalker = walker;
            mmu->numPagewalks++;
            pageWalker->translateTiming(req, tc, this, mode);
        } */
    };

	std::map<Addr, std::list<L2TranslationRequest*> > outstandingL2TLBreqs;
    //Sharmila- should be something other than TranslationRequest

	std::map<Addr, GPUPBEntry> L1prefetchBuffer;
	int L1prefetchBufferSize;
	//int prefetchAheadDistance;
	uint32_t id; //Sharmila- to identify which core's L1 TLB this is.
	uint32_t numcores; //Sharmila- to get the total number of shader cores

	std::vector<uint32_t> counters; //Sharmila- Instead of uint32_t, bit varying should be done
	
	void increment_counter(unsigned int TLB_ID, unsigned int core_counter);
	void decrement_counter(unsigned int TLB_ID, unsigned int core_counter);
	CudaGPU* getcudagpu() {return cudaGPU;}

};
#endif /* SHADER_TLB_HH_ */
