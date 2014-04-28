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

#include <map>

#include "arch/x86/insts/microldstop.hh"
#include "arch/x86/regs/misc.hh"
#include "arch/x86/regs/msr.hh"
#include "arch/x86/faults.hh"
#include "arch/x86/pagetable_walker.hh"
#include "debug/ShaderTLB.hh"
#include "gpu/shader_tlb.hh"
#include "gpu/gpgpu-sim/cuda_gpu.hh"

#define THRESHOLD 8

using namespace std;
using namespace X86ISA;

ShaderTLB::ShaderTLB(const Params *p) :
    BaseTLB(p), numEntries(p->entries), hitLatency(p->hit_latency),
    cudaGPU(p->gpu), accessHostPageTable(p->access_host_pagetable),
    L1prefetchBufferSize(p->L1prefetch_buffer_size), id(p->tlbid) //Sharmila
{
	DPRINTF(ShaderTLB, "ShaderTLB: Within ShaderTLB constructor, id = %d\n", id);
    if (numEntries > 0) {
        tlbMemory = new TLBMemory(p->entries, p->associativity);
    } else {
        tlbMemory = new InfiniteTLBMemory();
    }

    mmu = cudaGPU->getMMU();
    DPRINTF(ShaderTLB, "ShaderTLB: Linked mmu\n");

    // Sharmila- to get the number of shader cores
    numcores = cudaGPU->num_cores();
    DPRINTF(ShaderTLB, "ShaderTLB: Obtained num cores = %d\n", numcores);

    // Sharmila- Initialize all counters of this TLB to 0
    for(int i=0; i<numcores; i++){
    	counters[i] = 0;
    }
    // Sharmila- Adding this TLB into list of all TLBs
    //ShaderTLB *TLB = this;
    cudaGPU->L1TLBList[id] = this;
    DPRINTF(ShaderTLB, "ShaderTLB: Added this TLB to cudagpu L1TLB list\n");
    //TLB = this;
    DPRINTF(ShaderTLB, "ShaderTLB: Completed ShaderTLB constructor\n");
}

void
ShaderTLB::unserialize(Checkpoint *cp, const std::string &section)
{
    // Intentionally left blank to keep from trying to read shader header from
    // checkpoint files. Allows for restore into any number of shader cores.
    // NOTE: Cannot checkpoint during kernels
}


void
ShaderTLB::beginTranslateTiming(RequestPtr req,
                                BaseTLB::Translation *translation,
                                BaseTLB::Mode mode)
{
    if (accessHostPageTable) {
        translateTiming(req, cudaGPU->getThreadContext(), translation, mode);
    } else {
        // The below code implements a perfect TLB with instant access to the
        // device page table.
        // TODO: We can shift this around, maybe to memory, maybe hierarchical TLBs
        assert(numEntries == 0);
        Addr vaddr = req->getVaddr();
        Addr page_vaddr = cudaGPU->getGPUPageTable()->addrToPage(vaddr);
        Addr offset = vaddr - page_vaddr;
        Addr page_paddr;
        if (cudaGPU->getGPUPageTable()->lookup(page_vaddr, page_paddr)) {
            DPRINTF(ShaderTLB, "Translation found for vaddr %x = paddr %x\n",
                                vaddr, page_paddr + offset);
            req->setPaddr(page_paddr + offset);
            translation->finish(NoFault, req, NULL, mode);
        } else {
            panic("ShaderTLB missing translation for vaddr: %p! @pc: %p",
                    vaddr, req->getPC());
        }
    }
}

void
ShaderTLB::translateTiming(RequestPtr req, ThreadContext *tc,
                           Translation *translation, Mode mode)
{
    uint32_t flags = req->getFlags();

    // If this is true, we're dealing with a request to a non-memory address
    // space.
    if ((flags & SegmentFlagMask) == SEGMENT_REG_MS) {
        panic("GPU TLB cannot deal with non-memory addresses");
    }

    Addr vaddr = req->getVaddr();
    DPRINTF(ShaderTLB, "Translating vaddr %#x.\n", vaddr);

    HandyM5Reg m5Reg = tc->readMiscRegNoEffect(MISCREG_M5_REG);

    assert(m5Reg.prot); // Cannot deal with unprotected mode
    assert(m5Reg.mode == LongMode); // must be in long mode
    assert(m5Reg.submode == SixtyFourBitMode); // Assuming 64-bit mode
    assert(m5Reg.paging); // Paging better be enabled!

    Addr offset = vaddr % TheISA::PageBytes;
    Addr vpn = vaddr - offset;
    Addr ppn;

    if (tlbMemory->lookup(vpn, ppn)) {
        DPRINTF(ShaderTLB, "TLB hit. Phys addr %#x.\n", ppn + offset);
        L1TLBhits++;
        req->setPaddr(ppn + offset);
        translation->finish(NoFault, req, tc, mode);
    } else {
        // TLB miss! Let the x86 TLB handle the walk, etc
        DPRINTF(ShaderTLB, "TLB miss for addr %#x\n", vaddr);
        //Sharmila
        L1TLBmisses++;
        translation->markDelayed();

        //Sharmila
        //Look up in L1 prefetch buffer
        // Check for a hit in the prefetch buffers
		auto it = L1prefetchBuffer.find(vpn);
		if (it != L1prefetchBuffer.end()) {
			// Hit in the prefetch buffer
			L1prefetchHits++;
			ppn = it->second.ppn;
			//if (tlbMemory) {
			tlbMemory->insertL1TLB(this, vpn, ppn, true); //Sharmila-tlbMemory is either finite of infinite
			//}
			req->setPaddr(ppn + offset);

			// Sharmila
			// PB hit- Send message to leader TLB to increase this counter's value
			DPRINTF(ShaderTLB, "PB hit, sending message to leader TLB\n");
			uint32_t Source_TLB_ID = it->second.TLB_ID;

			// Get required ShaderTLB*
			auto it1 = cudaGPU->L1TLBList.find(Source_TLB_ID);
			ShaderTLB* Source_TLB = it1->second; //wrong
			Source_TLB->increment_counter(Source_TLB_ID, id);

			//req_tlb->insert(vpn, ppn);
			translation->finish(NoFault, req, tc, mode);
			// Remove from prefetchBuffer
			L1prefetchBuffer.erase(it);
			// This was a hit in the prefetch buffer, so we must have done the
			// right thing, Let's see if we get lucky again.
			//tryL1Prefetch(vpn, tc); //Sharmila-Not required for LF
			return;
		}
		//If miss in prefetch buffer, request should be sent to MMU
		else {
			// PB miss- Send message to all TLBs to increase this counter's value
			DPRINTF(ShaderTLB, "PB miss, sending message to all the TLBs\n");
			for(int i=0; i<numcores; i++)
			{
				if(i!=id)
				{
					// Get required ShaderTLB*
					auto it1 = cudaGPU->L1TLBList.find(i);
					ShaderTLB* Shader_TLB = it1->second;
					//ShaderTLB* Shader_TLB = cudaGPU->L1TLBList.find(i);
					Shader_TLB->increment_counter(i, id);
				}
			}
			//addition till here -Sharmila

			mmu->beginTLBMiss(this, translation, req, mode, tc);
			//Try prefetch on demand misses, but after sending the request to MMU
			//tryL1Prefetch(vpn, tc); //Sharmila-Not required for LF
		}
    }
}

void
ShaderTLB::insert(Addr vpn, Addr ppn)
{
    tlbMemory->insertL1TLB(this, vpn, ppn, true);
}

void
ShaderTLB::demapPage(Addr addr, uint64_t asn)
{
    DPRINTF(ShaderTLB, "Demapping %#x.\n", addr);
    panic("Demap addr unimplemented");
}

void
ShaderTLB::flushAll()
{
    panic("Flush all unimplemented");
}

bool
TLBMemory::lookup(Addr vpn, Addr& ppn, bool set_mru)
{
    int way = (vpn / TheISA::PageBytes) % ways;
    for (int i=0; i < sets; i++) {
        if (entries[way][i].vpn == vpn && !entries[way][i].free) {
            ppn = entries[way][i].ppn;
            assert(entries[way][i].mruTick > 0);
            if (set_mru) {
                entries[way][i].setMRU();
            }
            entries[way][i].hits++;
            return true;
        }
    }
    ppn = Addr(0);
    return false;
}

void
TLBMemory::insert(Addr vpn, Addr ppn)
{
    Addr a;
    if (lookup(vpn, a)) {
        return;
    }
    int way = (vpn / TheISA::PageBytes) % ways;
    GPUTlbEntry* entry = NULL;
    Tick minTick = curTick();
    for (int i=0; i < sets; i++) {
        if (entries[way][i].free) {
            entry = &entries[way][i];
            break;
        } else if (entries[way][i].mruTick < minTick) {
            minTick = entries[way][i].mruTick;
            entry = &entries[way][i];
        }
    }
    assert(entry);
    if (!entry->free) {
        DPRINTF(ShaderTLB, "Evicting entry for vpn %#x\n", entry->vpn);
    }

    entry->vpn = vpn;
    entry->ppn = ppn;
    entry->free = false;
    entry->setMRU();
}

//Sharmila- Use this prefetch function
void
TLBMemory::insertL1TLB(ShaderTLB *tlb, Addr vpn, Addr ppn, bool Prefetch)
{
    Addr a;
    if (lookup(vpn, a)) {
        return;
    }
    int way = (vpn / TheISA::PageBytes) % ways;
    GPUTlbEntry* entry = NULL;
    Tick minTick = curTick();
    for (int i=0; i < sets; i++) {
        if (entries[way][i].free) {
            entry = &entries[way][i];
            break;
        } else if (entries[way][i].mruTick < minTick) {
            minTick = entries[way][i].mruTick;
            entry = &entries[way][i];
        }
    }
    assert(entry);
    if (!entry->free) {
        DPRINTF(ShaderTLB, "Evicting entry for vpn %#x\n", entry->vpn);
    }

    entry->vpn = vpn;
    entry->ppn = ppn;
    entry->free = false;
    entry->setMRU();

    if(Prefetch)
    {
    	DPRINTF(ShaderTLB, "Inserting into other PBs while inserting into this TLB\n");
    	for(int it=0; it<tlb->numcores; it++)
    	{
    		if(tlb->counters[it] > THRESHOLD)
    		{
    			// Get required ShaderTLB*
    			auto it1 = tlb->getcudagpu()->L1TLBList.find(it);
    			ShaderTLB* follower = it1->second;
    			//ShaderTLB *follower = tlb->cudaGPU->L1TLBList.find(it);
    			//if(follower != tlb->getcudagpu()->L1TLBList.end())
    			follower->insertL1Prefetch(vpn, ppn, tlb->id);
    		}
    	}
    }
}

void
ShaderTLB::regStats()
{
    L1TLBhits
        .name(name()+".L1TLBhits")
        .desc("Number of hits in this TLB")
        ;
    L1TLBmisses
        .name(name()+".L1TLBmisses")
        .desc("Number of misses in L1 TLB")
        ;
    hitRate
        .name(name()+".hitRate")
        .desc("Hit rate for this TLB")
        ;

    hitRate = L1TLBhits / (L1TLBhits + L1TLBmisses);

    numL1Prefetches
            .name(name()+".numL1Prefetches")
            .desc("Number of L1 Prefetches")
            ;
    L1prefetchHits
            .name(name()+".L1prefetchHits")
            .desc("Hit rate of L1 Prefetch buffer for this TLB")
            ;
    PBhitRate
            .name(name()+".PBhitRate")
            .desc("Hit rate for this Prefetch Buffer")
            ;

    PBhitRate = L1prefetchHits / numL1Prefetches;
}

ShaderTLB *
ShaderTLBParams::create()
{
    return new ShaderTLB(this);
}

//Sharmila- This function not needed for LF PB
/*
void
ShaderTLB::tryL1Prefetch(Addr vpn, ThreadContext *tc)
{
    // If not using a prefetcher, skip this function.
    if (L1prefetchBufferSize == 0) {
        return;
    }

    // If this address has already been prefetched, skip
    auto it = L1prefetchBuffer.find(vpn);
    if (it != L1prefetchBuffer.end()) {
        return;
    }
    // Now send request to the MMU to fetch new address
    Addr next_vpn = vpn + TheISA::PageBytes;
    Addr ppn;
    if (tlbMemory->lookup(next_vpn, ppn, false)) { //Change the name
        // This vpn already in the TLB, no need to prefetch
        return;
    }

    //Sharmila
    if (outstandingL2TLBreqs.find(next_vpn) != outstandingL2TLBreqs.end()) {
        // Already walking for this vpn, no need to prefetch
        return;
    }

    numL1Prefetches++; //Sharmila

    // Prefetch the next PTE into the TLB.
    //Sharmila, set isPrefetch() true by sending PREFETCH flag

    Request::Flags flags = Request::PREFETCH;
    RequestPtr req = new Request(0, next_vpn, 4, flags, 0, 0, 0, 0);
    // Request(int asid, Addr vaddr, int size, Flags flags, MasterID mid, Addr pc, int cid, ThreadID tid)

    if(!req->isPrefetch())
    	return;

    L2TranslationRequest *L2translation = new L2TranslationRequest(this, NULL, //Should this be NULL or Translation?
    												req, BaseTLB::Read, tc, true);
    outstandingL2TLBreqs[next_vpn].push_back(L2translation);

    DPRINTF(ShaderTLB, "L1 Prefetching translation for %#x.\n", next_vpn);

    mmu->beginTLBMiss(this, NULL, req, BaseTLB::Read, tc); //Sharmila
    //ShaderMMU::beginTLBMiss(ShaderTLB *req_tlb, BaseTLB::Translation *translation, RequestPtr req, BaseTLB::Mode mode, ThreadContext *tc)
}
*/

void
ShaderTLB::insertL1Prefetch(Addr vpn, Addr ppn, uint32_t TLB_ID)
{
    DPRINTF(ShaderMMU, "Inserting %#x->%#x into pf buffer\n", vpn, ppn);
    assert(vpn % TheISA::PageBytes == 0);
    // Insert into prefetch buffer
    if (L1prefetchBuffer.size() >= L1prefetchBufferSize) {
        // evict unused entry from prefetch buffer
        auto min = L1prefetchBuffer.begin();
        Tick minTick = curTick();
        for (auto it=L1prefetchBuffer.begin(); it!=L1prefetchBuffer.end(); it++) {
            if (it->second.mruTick < minTick) {
                minTick = it->second.mruTick;
                min = it;
            }
        }
        assert(minTick != curTick() && min != L1prefetchBuffer.end());

        // Sharmila- PB evict, send message to leader TLB to decrement
        DPRINTF(ShaderTLB, "PB evict, sending message to Leader TLB to decrement counter value\n");
        uint32_t Leader_TLB_ID = min->second.TLB_ID;

        auto itt = cudaGPU->L1TLBList.find(Leader_TLB_ID);
        ShaderTLB* Leader_TLB = itt->second;
        Leader_TLB->decrement_counter(Leader_TLB_ID, this->id);
        L1prefetchBuffer.erase(min);
    }
    GPUPBEntry &e = L1prefetchBuffer[vpn];
    e.vpn = vpn;
    e.ppn = ppn;
    e.TLB_ID = TLB_ID;
    e.setMRU();
    assert(L1prefetchBuffer.size() <= L1prefetchBufferSize);
}

ShaderTLB::L2TranslationRequest::L2TranslationRequest(ShaderTLB *_tlb, BaseTLB::Translation *translation,
    RequestPtr _req, BaseTLB::Mode _mode, ThreadContext *_tc, bool prefetch)
            : TLB(_tlb), wrappedTranslation(translation), req(_req), mode(_mode), tc(_tc),
              beginFault(0), prefetch(prefetch)
{
    vpn = req->getVaddr() - req->getVaddr() % TheISA::PageBytes;
}

void
ShaderTLB::increment_counter(unsigned int TLB_ID, unsigned int core_counter)
{
	DPRINTF(ShaderTLB, "In Increment counter function, incrementing %d's %d counter\n", TLB_ID, core_counter);
	if(id == TLB_ID)
	{
		counters[core_counter]++;
	}
}

void
ShaderTLB::decrement_counter(unsigned int TLB_ID, unsigned int core_counter)
{
	DPRINTF(ShaderTLB, "In Decrement counter function, decrementing %d's %d counter\n", TLB_ID, core_counter);
	if(id == TLB_ID)
	{
		counters[core_counter]--;
	}
}
