// Dynamics capability routing plus A/C-grid acoustic dispatch families.
DynamicsComputeKernels* MetalComputeBackend::dynamics_kernels()
{
    return ready(nimbus::compute::KernelId::StateRelaxationCartesian) && ready(nimbus::compute::KernelId::SlowApply)
               ? this
               : nullptr;
}

const DynamicsComputeKernels* MetalComputeBackend::dynamics_kernels() const
{
    return ready(nimbus::compute::KernelId::StateRelaxationCartesian) && ready(nimbus::compute::KernelId::SlowApply)
               ? this
               : nullptr;
}

bool MetalComputeBackend::supports(DynamicsKernelOp op) const
{
    if (op == DynamicsKernelOp::AcousticContinuityMassFluxCartesianAgrid)
        return ready(nimbus::compute::KernelId::AcousticContinuityMassFluxCartesianAgrid);
    if (op == DynamicsKernelOp::AcousticPressureEosCartesianAgrid)
        return ready(nimbus::compute::KernelId::AcousticPressureEosCartesianAgrid);
    if (op == DynamicsKernelOp::AcousticMomentumCartesianAgrid)
        return ready(nimbus::compute::KernelId::AcousticMomentumCartesianAgrid);
    if (op == DynamicsKernelOp::AcousticNullModeConstraintCartesianAgrid)
        return ready(nimbus::compute::KernelId::AcousticNullModeConstraintCartesianAgrid);
    if (op == DynamicsKernelOp::AcousticBoundaryCartesianCgrid)
        return ready(nimbus::compute::KernelId::AcousticBcFaceCartesianCgrid);
    if (op == DynamicsKernelOp::AcousticBoundaryCartesianAgrid)
        return ready(nimbus::compute::KernelId::AcousticBoundaryCartesianAgrid);
    if (op == DynamicsKernelOp::StateRelaxationCartesian)
        return ready(nimbus::compute::KernelId::StateRelaxationCartesian);
    if (op == DynamicsKernelOp::SlowApply)
        return ready(nimbus::compute::KernelId::SlowApply);
    if (op == DynamicsKernelOp::ConservativeReduction)
        return ready(nimbus::compute::KernelId::ConservativeReduction);
    if (op == DynamicsKernelOp::ConservativeMomentum)
        return ready(nimbus::compute::KernelId::ConservativeMomentum);
    if (op == DynamicsKernelOp::CartesianCgridSlowTendencies)
        return ready(nimbus::compute::KernelId::CartesianCgridSlowTendencies);
    if (op == DynamicsKernelOp::CartesianTendencies)
        return ready(nimbus::compute::KernelId::CartesianTendencies);
    if (op == DynamicsKernelOp::AcousticPressure)
        return ready(nimbus::compute::KernelId::AcousticPressureCartesian);
    if (op == DynamicsKernelOp::AcousticMomentum)
        return ready(nimbus::compute::KernelId::AcousticMomentumCartesian);
    if (op == DynamicsKernelOp::AcousticSubstepFused || op == DynamicsKernelOp::AcousticSubstepsBatched)
        return ready(nimbus::compute::KernelId::AcousticPressureCartesian) &&
               ready(nimbus::compute::KernelId::AcousticMomentumCartesian);
    if (op == DynamicsKernelOp::AcousticPressureCartesianCgrid)
        return ready(nimbus::compute::KernelId::AcousticPressureCartesianCgrid);
    if (op == DynamicsKernelOp::AcousticMomentumCartesianCgrid)
        return ready(nimbus::compute::KernelId::AcousticMomentumCartesianCgrid);
    if (op == DynamicsKernelOp::AcousticSubstepFusedCartesianCgrid)
        return ready(nimbus::compute::KernelId::AcousticPressureCartesianCgrid) &&
               ready(nimbus::compute::KernelId::AcousticMomentumCartesianCgrid);
    if (op == DynamicsKernelOp::AcousticSubstepsBatchedCartesianCgrid)
        return ready(nimbus::compute::KernelId::AcousticPressureCartesianCgrid) &&
               ready(nimbus::compute::KernelId::AcousticMomentumCartesianCgrid) &&
               ready(nimbus::compute::KernelId::AcousticBcFaceCartesianCgrid);
    return false;
}
