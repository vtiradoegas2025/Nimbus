#pragma once

#include <string_view>
#include <vector>

namespace tmv::runtime
{

enum class ReadinessState
{
    EngineeringIncomplete,
    Ready,
    CampaignValidated,
    ExternallyReviewed,
    FullyQualified,
};

struct ReadinessItem
{
    std::string_view id;
    std::string_view area;
    std::string_view scope;
    ReadinessState  state;
    std::string_view bounded_evidence;
    std::string_view campaign_evidence;
    std::string_view external_review_evidence;
    std::string_view blocker;
};

const char* readiness_state_name(ReadinessState state) noexcept;
const std::vector<ReadinessItem>& tracked_readiness_items();

} // namespace tmv::runtime
