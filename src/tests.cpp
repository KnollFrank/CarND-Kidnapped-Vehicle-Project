#include "gtest/gtest.h"
#include "particle_filter.h"

TEST(ParticleFilterTest, ShouldInit) {
  // GIVEN
  ParticleFilter pf;
  double sigma_pos[3] = {0.3, 0.3, 0.01};
  double x = 10.1;
  double y = 12.3;
  double theta = 45.7;

  ASSERT_FALSE(pf.initialized());

  // WHEN
  pf.init(x, y, theta, sigma_pos);

  // THEN
  ASSERT_EQ(50, pf.particles.size());
  for (int i = 0; i < pf.particles.size(); ++i) {
    EXPECT_NEAR(x, pf.particles[i].x, 3*sigma_pos[0]) << "particles.x at index " << i;
    EXPECT_NEAR(y, pf.particles[i].y, 3*sigma_pos[1]) << "particles.y at index " << i;
    EXPECT_NEAR(theta, pf.particles[i].theta, 3*sigma_pos[2]) << "particles.theta at index " << i;
  }

  ASSERT_EQ(pf.particles.size(), pf.weights.size());
  for (int i = 0; i < pf.weights.size(); ++i) {
    EXPECT_EQ(1, pf.weights[i]) << "weights should be 1 at index " << i;
  }

  ASSERT_TRUE(pf.initialized());
}

TEST(ParticleFilterTest, ShouldPredict) {
  // GIVEN
  ParticleFilter pf;
  double sigma_pos[3] = {0.0, 0.0, 0.0};
  pf.init(0, 0, 0, sigma_pos);

  // WHEN
  double delta_t = 1.0;
  double velocity = 1.0;
  double yaw_rate = M_PI/2.0;
  pf.prediction(delta_t, sigma_pos, velocity, yaw_rate);

  // THEN
  EXPECT_NEAR(0.0, pf.particles[0].x, 0.00001) << "particles[0].x";
  EXPECT_NEAR(1.0, pf.particles[0].y, 0.00001) << "particles[0].y";
  EXPECT_NEAR(M_PI/2.0, pf.particles[0].theta, 0.00001) << "particles[0].theta";
}
