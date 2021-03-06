# Copyright (c) 2006-2007 The Regents of The University of Michigan
# Copyright (c) 2009 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Authors: Brad Beckmann

import math
import m5
from m5.objects import *
from m5.defines import buildEnv

#
# Note: the L1 Cache latency is only used by the sequencer on fast path hits
#
class L1Cache(RubyCache):
    latency = 1

#
# Note: the L2 Cache latency is not currently used
#
class L2Cache(RubyCache):
    latency = 15

def create_system(options, system, piobus, dma_devices, ruby_system):

    if not buildEnv['GPGPU_SIM']:
        m5.util.panic("This script requires GPGPU-Sim integration to be built.")

    print "Creating system for GPU"

    # Run the original protocol script
    buildEnv['PROTOCOL'] = buildEnv['PROTOCOL'][:-7]
    protocol = buildEnv['PROTOCOL']
    exec "import %s" % protocol
    try:
        (cpu_sequencers, dir_cntrls, topology) = \
            eval("%s.create_system(options, system, piobus, dma_devices, ruby_system)" % protocol)
    except:
        print "Error: could not create system for ruby protocol inside fusion system %s" % protocol
        raise

    #
    # The ruby network creation expects the list of nodes in the system to be
    # consistent with the NetDest list.  Therefore the l1 controller nodes must be
    # listed before the directory nodes and directory nodes before dma nodes, etc.
    #
    l1_cntrl_nodes = []

    #
    # Caches for the stream processors
    #
    l2_bits = int(math.log(options.num_l2caches, 2))
    block_size_bits = int(math.log(options.cacheline_size, 2))

    cntrl_count = 0

    for i in xrange(options.num_sc):
        #
        # First create the Ruby objects associated with this cpu
        #
        l1i_cache = L1Cache(size = options.sc_l1_size,
                            assoc = options.sc_l1_assoc,
                            replacement_policy = "LRU",
                            start_index_bit = block_size_bits)
        l1d_cache = L1Cache(size = options.sc_l1_size,
                            assoc = options.sc_l1_assoc,
                            replacement_policy = "LRU",
                            start_index_bit = block_size_bits)

        prefetcher = RubyPrefetcher.Prefetcher()

        l1_cntrl = L1Cache_Controller(version = options.num_cpus + i,
                                      cntrl_id = len(topology),
                                      L1Icache = l1i_cache,
                                      L1Dcache = l1d_cache,
                                      l2_select_num_bits = l2_bits,
                                      send_evictions = (
                                          options.cpu_type == "detailed"),
                                      prefetcher = prefetcher,
                                      ruby_system = ruby_system,
                                      enable_prefetch = False)

        cpu_seq = RubySequencer(version = options.num_cpus + i,
                                icache = l1i_cache,
                                dcache = l1d_cache,
                                access_phys_mem = True,
                                max_outstanding_requests = options.gpu_l1_buf_depth,
                                ruby_system = ruby_system)

        l1_cntrl.sequencer = cpu_seq

        if piobus != None:
            cpu_seq.pio_port = piobus.slave

        exec("ruby_system.l1_cntrl_sp%02d = l1_cntrl" % i)

        #
        # Add controllers and sequencers to the appropriate lists
        #
        cpu_sequencers.append(cpu_seq)
        topology.addController(l1_cntrl)

        cntrl_count += 1

    ############################################################################
    # Pagewalk cache
    # NOTE: We use a CPU L1 cache controller here. This is to facilatate MMU
    #       cache coherence (as the GPU L1 caches are incoherent without flushes
    #       The L2 cache is small, and should have minimal affect on the
    #       performance (see Section 6.2 of Power et al. HPCA 2014).
    pwd_cache = L1Cache(size = options.pwc_size,
                            assoc = 16, # 64 is fully associative @ 8kB
                            replacement_policy = "LRU",
                            start_index_bit = block_size_bits,
                            latency = 8,
                            resourceStalls = False)
    # Small cache since CPU L1 requires I and D
    pwi_cache = L1Cache(size = "512B",
                            assoc = 2,
                            replacement_policy = "LRU",
                            start_index_bit = block_size_bits,
                            latency = 8,
                            resourceStalls = False)

    prefetcher = RubyPrefetcher.Prefetcher()

    l1_cntrl = L1Cache_Controller(version = options.num_cpus + options.num_sc,
                                  cntrl_id = len(topology),
                                  send_evictions = False,
                                  L1Icache = pwi_cache,
                                  L1Dcache = pwd_cache,
                                  l2_select_num_bits = l2_bits,
                                  prefetcher = prefetcher,
                                  ruby_system = ruby_system,
                                  enable_prefetch = False)

    cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc,
                            icache = pwd_cache, # Never get data from pwi_cache
                            dcache = pwd_cache,
                            access_phys_mem = True,
                            max_outstanding_requests = options.gpu_l1_buf_depth,
                            ruby_system = ruby_system,
                            deadlock_threshold = 2000000)

    l1_cntrl.sequencer = cpu_seq


    ruby_system.l1_pw_cntrl = l1_cntrl
    cpu_sequencers.append(cpu_seq)

    topology.addController(l1_cntrl)


    # Copy engine cache (make as small as possible, ideally 0)
    l1i_cache = L1Cache(size = "2kB", assoc = 2)
    l1d_cache = L1Cache(size = "2kB", assoc = 2)

    prefetcher = RubyPrefetcher.Prefetcher()

    l1_cntrl = L1Cache_Controller(version = options.num_cpus + options.num_sc+1,
                                  cntrl_id = len(topology),
                                  send_evictions = (
                                      options.cpu_type == "detailed"),
                                  L1Icache = l1i_cache,
                                  L1Dcache = l1d_cache,
                                  l2_select_num_bits = l2_bits,
                                  prefetcher = prefetcher,
                                  ruby_system = ruby_system,
                                  enable_prefetch = False)

    #
    # Only one unified L1 cache exists.  Can cache instructions and data.
    #
    cpu_seq = RubySequencer(version = options.num_cpus + options.num_sc + 1,
                            icache = l1i_cache,
                            dcache = l1d_cache,
                            access_phys_mem = True,
                            max_outstanding_requests = 64,
                            ruby_system = ruby_system)

    l1_cntrl.sequencer = cpu_seq

    if piobus != None:
        cpu_seq.pio_port = piobus.slave

    ruby_system.l1_cntrl_ce = l1_cntrl

    cpu_sequencers.append(cpu_seq)
    topology.addController(l1_cntrl)

    return (cpu_sequencers, dir_cntrls, topology)
