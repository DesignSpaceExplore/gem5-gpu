#!/bin/bash

FILEmmu=shader_mmu
FILEtlb=shader_tlb
FILElsq=shader_lsq
FILEcc=cuda_core
FILEcg=cuda_gpu
FILEggs=gpgpu-sim

rm -f $FILEmmu.cc $FILEmmu.hh $FILEtlb.cc $FILEtlb.hh $FILElsq.cc $FILElsq.hh $FILEggs/$FILEcc.cc $FILEggs/$FILEcc.hh $FILEggs/$FILEcg.cc $FILEggs/$FILEcg.hh
ln -s $FILEmmu$1.cc $FILEmmu.cc
ln -s $FILEmmu$1.hh $FILEmmu.hh
ln -s $FILEtlb$1.cc $FILEtlb.cc
ln -s $FILEtlb$1.hh $FILEtlb.hh
ln -s $FILElsq$1.cc $FILElsq.cc
ln -s $FILElsq$1.hh $FILElsq.hh
ln -s $FILEcc$1.cc $FILEggs/$FILEcc.cc
ln -s $FILEcc$1.hh $FILEggs/$FILEcc.hh
ln -s $FILEcg$1.cc $FILEggs/$FILEcg.cc
ln -s $FILEcg$1.hh $FILEggs/$FILEcg.hh
