#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @file common.hpp
 * @brief Forward declarations and callbacks shared by compute dispatch families.
 */

class Field3D;
struct CartesianCGridAcousticEnergyReportV1;
namespace tmv
{
class CartesianCGridTransportContext;
struct ScalarBoundaryTransportReport;
class CartesianInterfaceFluxRegister;
enum class ScalarMoistEnergyKind : std::uint8_t;
struct ScalarMoistEnergyTransportReportV1;
}

using FourFieldCompletedObserver = void (*)(
    void* context,
    const float* field0_committed,
    const float* field1_committed,
    const float* field2_committed,
    const float* field3_committed);

using TwoFieldCompletedObserver = void (*)(
    void* context,
    const float* field0_committed,
    const float* field1_committed);

using ThreeFieldCompletedObserver = void (*)(
    void* context,
    const float* field0_committed,
    const float* field1_committed,
    const float* field2_committed);

using FiveFieldCompletedObserver = void (*)(
    void* context,
    const float* field0_committed,
    const float* field1_committed,
    const float* field2_committed,
    const float* field3_committed,
    const float* field4_committed);

using SixFieldCompletedObserver = void (*)(
    void* context,
    const float* field0_committed,
    const float* field1_committed,
    const float* field2_committed,
    const float* field3_committed,
    const float* field4_committed,
    const float* field5_committed);
