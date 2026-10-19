#pragma once

#include "core/field/field3d.hpp"

#include <vector>

// C-grid output uses cell-centered velocity. Interior cells average adjacent
// faces; west/south use their maintained boundary face. The two-argument w
// overload is the flat-ground contract (lower face w=0). Terrain callers must
// provide the physical terrain-tangent lower-face velocity for every
// horizontal source column. Call only for staggered velocity fields.
// Destination dimensions are derived from the source, not process-global grid.

void interpolate_u_face_to_center(const Field3D& u_face, Field3D& u_center);
void interpolate_v_face_to_center(const Field3D& v_face, Field3D& v_center);
void interpolate_w_face_to_center(const Field3D& w_face, Field3D& w_center);
void interpolate_w_face_to_center(const Field3D&             w_face,
                                  Field3D&                   w_center,
                                  const std::vector<double>& lower_boundary_w);
