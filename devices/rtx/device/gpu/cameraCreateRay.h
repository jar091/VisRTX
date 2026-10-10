/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "gpu/halton.h"
#include "gpu_objects.h"

namespace visrtx {

VISRTX_DEVICE Ray cameraCreateRay(const CameraGPUData &c, vec2 screen, vec2 r)
{
  Ray ray;

  screen.x = glm::mix(c.region[0], c.region[2], screen.x);
  screen.y = glm::mix(c.region[1], c.region[3], screen.y);

  switch (c.type) {
  case CameraType::PERSPECTIVE: {
    const auto &p = c.perspective;
    ray.org = c.pos;
    ray.dir = p.dir_00 + screen.x * p.dir_du + screen.y * p.dir_dv;

    if (p.scaledAperture > 0.f) {
      const vec2 llp = uniformSampleDisk(p.scaledAperture, r);
      const vec3 lp = (llp.x * p.dir_du) + ((llp.y * p.aspect) * p.dir_dv);
      ray.org += lp;
      ray.dir -= lp;
    }

    ray.dir = normalize(ray.dir);
    break;
  }
  case CameraType::ORTHOGRAPHIC: {
    const auto &o = c.orthographic;
    ray.dir = c.dir;
    ray.org = o.pos_00 + screen.x * o.pos_du + screen.y * o.pos_dv;
    break;
  }
  default:
    break;
  }

  return ray;
}

// Clips a camera ray to the slab between the near and far clip planes of the
// camera. The planes are perpendicular to the camera direction: their
// distances are measured along that direction, not along the ray. They are
// intersected in world space, so 'ray.org' can be any point of the camera ray
// (lens sample, origin advanced by a cutting plane, continuation behind a
// see-through surface). Only 'ray.t' changes: hit distances stay relative to
// 'ray.org', and rays other than camera rays are never clipped.
VISRTX_DEVICE void applyCameraClipPlanes(const CameraGPUData &c, Ray &ray)
{
  const bool clipNear = c.clipNear > 0.f;
  const bool clipFar = c.clipFar < std::numeric_limits<float>::max();
  if (!clipNear && !clipFar)
    return;

  const float cosAxis = glm::dot(ray.dir, c.dir);
  if (cosAxis > 1e-6f) {
    const float axisDist = glm::dot(ray.org - c.pos, c.dir);
    if (clipNear)
      ray.t.lower = glm::max(ray.t.lower, (c.clipNear - axisDist) / cosAxis);
    if (clipFar)
      ray.t.upper = glm::min(ray.t.upper, (c.clipFar - axisDist) / cosAxis);
  } else {
    ray.t.upper = 0.f; // ray never enters the slab
  }

  if (!(ray.t.lower <= ray.t.upper))
    ray.t = box1(0.f, 0.f); // nothing of the ray is left
}

// sampleIdx is a (frame-counter × spp_loop) ordinal advancing once per
// camera sample within the frame's accumulation; combined with a
// per-pixel hash offset it indexes a Halton 4D point used for
// sub-pixel jitter (dims 0/1) + lens aperture (dims 2/3). Halton at
// the camera level converges the AA / DoF integrals strictly faster
// than uniform random — see gpu/halton.h.
VISRTX_DEVICE Ray makePrimaryRay(
    ScreenSample &ss, uint32_t sampleIdx, bool centerPixel = false)
{
  const uint32_t haltonIdx =
      sampleIdx + haltonPixelHash(ss.pixel.x, ss.pixel.y);
  const ::float4 r = halton4D(haltonIdx);
  ss.screen = (centerPixel ? vec2(ss.pixel.x + 0.5f, ss.pixel.y + 0.5f)
                           : vec2(ss.pixel.x + r.x, ss.pixel.y + r.y))
      * ss.frameData->fb.invSize;
  return cameraCreateRay(ss.frameData->camera, ss.screen, {r.z, r.w});
}

} // namespace visrtx
