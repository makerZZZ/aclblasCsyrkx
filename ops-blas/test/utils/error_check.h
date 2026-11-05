#pragma once

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define CHECK_ACLRT(func)                                                               \
  {                                                                                     \
    aclError status = (func);                                                           \
    if (status != ACL_SUCCESS) {                                                        \
      std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

#define CHECK_ACLBLAS(func)                                                             \
  {                                                                                     \
    aclblasStatus_t status = (func);                                                    \
    if (status != ACLBLAS_STATUS_SUCCESS) {                                             \
      std::cerr << "BLASLT Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

