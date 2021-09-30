/*************************************************************************
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"
#include "collectives.h"

// QUESTION : 实在是不知道这些pattern是啥意思，很模糊
typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollTreeUpDown
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  ncclDevRedOpFull opFull;
  
  // READNOTE : 在 getAlgoInfo 中计算
  // READNOTE : 在 getPatternInfo 中计算
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;

  // READNOTE : 在 getLoopInfo 中计算
  // 其中 nstepsPerLoop 和 nstepsPerLoop 是根据pattern计算出来的
  // nBytes = info->count*ncclTypeSize(info->datatype)
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
  ssize_t sendbytes;
  ssize_t recvbytes;
  int recvChunkSize;
  int sendChunkSize;
  uint32_t delta;
  int channelId;
};

#endif
