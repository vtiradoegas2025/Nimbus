#pragma once

#include <memory>
#include <string>
#include "core/infra/domain_motion.hpp"

struct SimulationStorageState;
struct SimulationStateView;

namespace tmv
{

// The caller validates scientific outputs before sealing. Publication then
// transfers all owners together and rebinds execution references to the live run.
class EarthFixedStepTransaction
{
public:
    EarthFixedStepTransaction(SimulationStorageState& live, double dt_s);
    ~EarthFixedStepTransaction();
    EarthFixedStepTransaction(const EarthFixedStepTransaction&) = delete;
    EarthFixedStepTransaction& operator=(const EarthFixedStepTransaction&) = delete;

    SimulationStateView state();
    void finish_evolution();
    void prepare_commit();
    bool publish() noexcept;
    void publish_reports();

private:
    SimulationStorageState& live_;
    std::unique_ptr<SimulationStorageState> candidate_;
    double begin_time_s_;
    double end_time_s_;
    domain::HorizontalDomainFrame frame_;
    struct Reports;
    std::unique_ptr<Reports> reports_;
    bool moving_ = false;
    bool evolved_ = false;
    bool prepared_ = false;
    bool published_ = false;
};

void open_earth_fixed_remap_csv(const std::string& path);
void close_earth_fixed_remap_csv();

} // namespace tmv
