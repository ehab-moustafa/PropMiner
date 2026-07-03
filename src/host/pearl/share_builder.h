#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "pearl_mining_wrapper.h"
#include "pearl_types.h"

namespace pearl {

class SigmaContext;

// Constructs the Pearl V2 protobuf ShareSubmission from a ShareFound event.
class ShareBuilder {
public:
    ShareBuilder(const MiningConfig& cfg);

    // Build the full protobuf-encoded ShareSubmission for gRPC submission.
    std::vector<uint8_t> build(const ShareFound& share,
                                const SigmaContext& ctx) const;

    // Verify a built proof locally (for tests).
    bool verify(const std::vector<uint8_t>& proof,
                const SigmaContext& ctx) const;

    // High-level share correctness check: rebuild A/B proofs and recompute the
    // claimed hash.  Requires the share to carry a_slice and a_leaf_cvs.
    static bool VerifyShare(const ShareFound& share, const SigmaContext& ctx);

    // Public test hook: recompute the claimed hash from opened rows/cols.
    static std::array<uint8_t, 32> ComputeClaimedHash(
        const ShareFound& share,
        const uint8_t job_key[32],
        const uint8_t hashA[32],
        const uint8_t hashB[32]);

private:
    MiningConfig cfg_;
    MiningCapi mining_;

    // Derive noise seeds exactly like CommitmentHasher.DeriveNoiseSeeds.
    static void derive_noise_seeds(const uint8_t job_key[32],
                                   const uint8_t hashA[32],
                                   const uint8_t hashB[32],
                                   uint8_t b_noise_seed[32],
                                   uint8_t a_noise_seed[32]);

    // Compute jackpot transcript and hash it.
    std::array<uint8_t, 32> compute_claimed_hash(
        const ShareFound& share,
        const uint8_t job_key[32],
        const uint8_t hashA[32],
        const uint8_t hashB[32]) const;

    // Derive audit leaf indices per AuditIndexDeriver.Derive.
    std::vector<uint32_t> derive_audit_indices(
        const uint8_t claimed_hash[32],
        const uint8_t b_seed[32],
        uint32_t audit_k,
        uint32_t total_leaves) const;
};

} // namespace pearl
