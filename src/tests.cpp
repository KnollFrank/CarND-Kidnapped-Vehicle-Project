#include <gtest/gtest.h>
#include "particle_filter.h"

TEST(ParticleFilterTest, Bla) {
  ParticleFilter pf;
  ASSERT_EQ(false, pf.initialized());
  double sigma_pos[3] = { 0.3, 0.3, 0.01 };
  double sense_x = 10.1;
  double sense_y = 12.3;
  double sense_theta = 45.7;
  pf.init(sense_x, sense_y, sense_theta, sigma_pos);
  ASSERT_EQ(true, pf.initialized());
}
