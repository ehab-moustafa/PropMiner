#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t pearl_capi_version(void);

void pearl_capi_free_buffer(uint8_t* ptr, size_t len);
void pearl_capi_free_u32_buffer(uint32_t* ptr, size_t len);
void pearl_capi_free_string(char* ptr);

int pearl_capi_blake3_keyed(
    const uint8_t* data, size_t data_len,
    const uint8_t* key, uint8_t* out,
    char** err_msg);

int pearl_capi_blake3_hash(
    const uint8_t* data, size_t data_len,
    uint8_t* out,
    char** err_msg);

// Write exactly out_len bytes of BLAKE3 XOF output from data.
int pearl_capi_blake3_xof(
    const uint8_t* data, size_t data_len,
    uint8_t* out, size_t out_len,
    char** err_msg);

int pearl_capi_merkle_root(
    const uint8_t* data, size_t data_len,
    const uint8_t* key, uint8_t* out_root,
    char** err_msg);

int pearl_capi_bseed_expand_and_merkle(
    const uint8_t* bseed,
    size_t n, size_t k,
    const uint8_t* key,
    uint8_t* out_root,
    char** err_msg);

int pearl_capi_bseed_expand_raw(
    const uint8_t* bseed,
    size_t n, size_t k,
    uint8_t* out_b, size_t out_b_len,
    char** err_msg);

int pearl_capi_bseed_expand_range_raw(
    const uint8_t* bseed,
    uint64_t byte_offset,
    uint8_t* out_b, size_t out_b_len,
    char** err_msg);

int pearl_capi_merkle_verify_proof(
    const uint8_t* leaf_data,
    const uint32_t* leaf_indices,
    size_t leaf_count,
    size_t total_leaves,
    const uint8_t* siblings,
    size_t sibling_count,
    const uint8_t* key,
    const uint8_t* expected_root,
    char** err_msg);

int pearl_capi_merkle_root_and_proof(
    const uint8_t* data, size_t data_len,
    const uint8_t* key,
    const uint32_t* row_indices, size_t row_indices_len,
    size_t row_width,
    uint8_t* out_root,
    uint32_t* out_total_leaves,
    uint8_t** out_leaf_data, size_t* out_leaf_count,
    uint32_t** out_leaf_indices, size_t* out_leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_count,
    char** err_msg);

// Merkle-tree handle API (build once, prove many).
int pearl_capi_merkle_build_tree(
    const uint8_t* data, size_t data_len,
    const uint8_t* key,
    size_t row_width,
    void** out_handle,
    uint8_t* out_root,
    uint32_t* out_total_leaves,
    char** err_msg);

int pearl_capi_merkle_proof_for_handle(
    void* handle,
    const uint32_t* row_indices, size_t row_indices_len,
    uint8_t** out_leaf_data, size_t* out_leaf_count,
    uint32_t** out_leaf_indices, size_t* out_leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_count,
    char** err_msg);

int pearl_capi_merkle_audit_paths_for_handle(
    void* handle,
    const uint32_t* leaf_indices, size_t leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_bytes,
    char** err_msg);

void pearl_capi_merkle_tree_free(void* handle);

// BSeed-backed tree from GPU leaf CVs.
int pearl_capi_bseed_merkle_build_tree_from_leaf_cvs(
    const uint8_t* leaf_cvs, size_t leaf_cvs_len,
    const uint8_t* bseed,
    const uint8_t* key,
    size_t num_rows, size_t row_width,
    void** out_handle,
    uint8_t* out_root,
    uint32_t* out_total_leaves,
    char** err_msg);

int pearl_capi_bseed_merkle_proof_for_handle(
    void* handle,
    const uint32_t* row_indices, size_t row_indices_len,
    uint8_t** out_leaf_data, size_t* out_leaf_count,
    uint32_t** out_leaf_indices, size_t* out_leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_count,
    char** err_msg);

int pearl_capi_bseed_merkle_audit_paths_for_handle(
    void* handle,
    const uint32_t* leaf_indices, size_t leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_bytes,
    char** err_msg);

void pearl_capi_bseed_merkle_tree_free(void* handle);

// A-side proof from leaf CVs.
int pearl_capi_merkle_proof_from_leaf_cvs(
    const uint8_t* leaf_cvs, size_t leaf_cvs_len,
    const uint8_t* leaf_data, size_t leaf_data_len,
    const uint8_t* key,
    const uint32_t* row_indices, size_t row_indices_len,
    size_t num_rows, size_t row_width,
    uint8_t* out_root,
    uint32_t* out_total_leaves,
    uint32_t** out_leaf_indices, size_t* out_leaf_indices_len,
    uint8_t** out_siblings, size_t* out_sibling_count,
    char** err_msg);

#ifdef __cplusplus
}
#endif
