#!/bin/bash

FILEmmu=shader_mmu
FILEtlb=shader_tlb

rm -f $FILEmmu.cc $FILEmmu.hh $FILEtlb.cc $FILEtlb.hh 
ln -s $FILEmmu$1.cc $FILEmmu.cc
ln -s $FILEmmu$1.hh $FILEmmu.hh
ln -s $FILEtlb$1.cc $FILEtlb.cc
ln -s $FILEtlb$1.hh $FILEtlb.hh
